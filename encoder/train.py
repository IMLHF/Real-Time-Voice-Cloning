from encoder.visualizations import Visualizations
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import speakers_per_batch, utterances_per_speaker, learning_rate_init, dataloader_workers
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path
import torch
import time
import sys

def sync(device: torch.device):
    # FIXME
    return
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
       torch.cuda.synchronize(device)

def train(run_id: str, train_data_root: Path, test_data_root: Path, models_dir: Path, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool):
    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(train_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=dataloader_workers,
        # pin_memory=True,
    )
    test_dataset = SpeakerVerificationDataset(test_data_root)
    testdata_loader = SpeakerVerificationDataLoader(
        test_dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=dataloader_workers,
        # pin_memory=True,
    )

    # Setup the device on which to run the forward pass and the loss. These can be different,
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model and the optimizer
    model = SpeakerEncoder(device)
    raw_model = model
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs.")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1

    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(str(state_fpath))
            init_step = checkpoint["step"]
            raw_model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()
    sys.stdout.flush()

    save_interval_s_time = time.time()
    prt_interval_s_time = time.time()
    total_loss, total_eer = 0, 0
    # Training loop
    profiler = Profiler(summarize_every=1, disabled=True)
    for step, speaker_batch in enumerate(loader, init_step):
        # step_s_time = time.time()
        sync(device)
        profiler.tick("Blocking, waiting for batch (threaded)")

        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("Data to %s" % device)
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass")
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1))
        loss, eer = raw_model.loss(embeds_loss)
        # print(loss.item(), flush=True)
        total_loss += loss.item()
        total_eer += eer
        sync(device)
        profiler.tick("Loss")

        # Backward pass
        model.zero_grad()
        loss.backward()
        profiler.tick("Backward pass")
        raw_model.do_gradient_ops()
        optimizer.step()
        sync(device)
        profiler.tick("Parameter update")

        if step % vis_every == 0:
            learning_rate = optimizer.param_groups[0]["lr"]
            prt_interval_e_time = time.time()
            cost_time = prt_interval_e_time - prt_interval_s_time
            prt_interval_s_time = prt_interval_e_time
            print("    Step %06d> %d step cost %d seconds, lr:%.4f, Avg_loss:%.4f, Avg_eer:%.4f." % (
                    #   step, save_every, cost_time, loss.detach().numpy(), eer), flush=True)
                    step, vis_every, cost_time, learning_rate, total_loss/vis_every, total_eer/vis_every), flush=True)
            total_loss, total_eer = 0, 0


        # Overwrite the latest version of the model && test model
        # save_every = 20
        if save_every != 0 and step % save_every == 0:
            # save
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                }, str(state_fpath))

            # test
            test_total_loss, test_total_eer = 0.0, 0.0
            for test_step, test_batch in enumerate(testdata_loader, 1):
                testinputs = torch.from_numpy(test_batch.data).to(device)
                with torch.no_grad():
                    test_embeds = model(testinputs)
                    test_embeds_loss = test_embeds.view((speakers_per_batch, utterances_per_speaker, -1))
                    test_loss, test_eer = raw_model.loss(test_embeds_loss)
                # print(loss.item(), flush=True)
                test_total_loss += test_loss.item()
                test_total_eer += test_eer
                test_prt_interval = 10
                if test_step % test_prt_interval == 0:
                    print("    |--Test Step %06d> Avg_loss:%.4f, Avg_eer:%.4f." % (
                        test_step, test_total_loss/test_step, test_total_eer/test_step), flush=True)
                if test_step == 50:
                    break

            # print log
            save_interval_e_time = time.time()
            cost_time = save_interval_e_time - save_interval_s_time
            print("\n"
                  "++++Step %06d> Saving the model, %d step cost %d seconds." % (
                    #   step, save_every, cost_time, loss.detach().numpy(), eer), flush=True)
                    step, save_every, cost_time), flush=True)
            save_interval_s_time = save_interval_e_time

        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            print("Making a backup (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = str(backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step)))
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
        sync(device)
        profiler.tick("Extras (visualizations, saving)")
        # step_e_time = time.time()
        # print("step loss:", loss.detach().numpy(), "step eer:", eer, flush=True)
        # print("step %06d> loss:%.4f, eer:%.4f, cost_time:%dms." % (
        #     step, loss.detach().numpy(), eer, (step_e_time-step_s_time)*1000),
        #     flush=True)

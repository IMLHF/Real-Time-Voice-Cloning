from encoder.preprocess import preprocess_librispeech_other, preprocess_voxceleb1, preprocess_voxceleb2
from encoder.preprocess import preprocess_SLR68, preprocess_SLR38, preprocess_aishell2, preprocess_libri_test
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms and "
                    "writes them to the disk. This will allow you to train the encoder. The "
                    "datasets required are at least one of VoxCeleb1, VoxCeleb2 and LibriSpeech. "
                    "Ideally, you should have all three. You should extract them as they are "
                    "after having downloaded them and put them in a same directory, e.g.:\n"
                    "-[datasets_root]\n"
                    "  -LibriSpeech\n"
                    "    -train-other-500\n"
                    "  -VoxCeleb1\n"
                    "    -wav\n"
                    "    -vox1_meta.csv\n"
                    "  -VoxCeleb2\n"
                    "    -dev",
        formatter_class=MyFormatter
    )
    parser.add_argument("--datasets_root", type=Path, required=True,
                        help="Path to the directory containing your LibriSpeech/TTS, VoxCeleb datasets and SLR68.")
    parser.add_argument("-d", "--datasets", type=str,
                        required=True,
                        # default="librispeech_other,voxceleb1,voxceleb2,SLR68,SLR38", 
                        help="Comma-separated list of the name of the datasets you want to preprocess. Only the train "
                        "set of these datasets will be used. Possible names: librispeech_other, voxceleb1, "
                        "voxceleb2, SLR68, SLR38.")
    parser.add_argument("-o", "--out_dir", type=Path, default="encoder_train",
                        help="Path to the output directory that will contain the mel spectrograms."
                        "Defaults to <datasets_root>/SV2TTS/encoder_train/")
    parser.add_argument("-s", "--skip_existing", action="store_true",
                        help="Whether to skip existing output files with the same name. Useful if this script was "
                        "interrupted.")
    args = parser.parse_args()

    # Process the arguments
    args.datasets = args.datasets.split(",")
    args.out_dir = args.datasets_root.joinpath("SV2TTS", args.out_dir)
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the datasets
    print_args(args, parser)
    preprocess_func = {
        "librispeech_other": preprocess_librispeech_other,
        "librispeech_test": preprocess_libri_test,
        "voxceleb1": preprocess_voxceleb1,
        "voxceleb2": preprocess_voxceleb2,
        "SLR68": preprocess_SLR68,
        "SLR38": preprocess_SLR38,
        "aishell2": preprocess_aishell2,
    }
    args = vars(args)
    for dataset in args.pop("datasets"):
        print("Preprocessing %s" % dataset)
        preprocess_func[dataset](**args)

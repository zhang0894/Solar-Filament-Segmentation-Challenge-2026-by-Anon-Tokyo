"""Signal only the recorded training parent; loader workers remain intact."""

import argparse
import json

from filament.runtime import stop_training_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    print(json.dumps(stop_training_process(args.run)))

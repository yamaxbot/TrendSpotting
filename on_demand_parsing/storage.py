import json
import os


# Папка, внутри которой будут создаваться output1, output2, ...
OUTPUT_DIR = r""


def create_output_dir():
    number = 1

    while True:
        output_dir = os.path.join(
            OUTPUT_DIR,
            f"output{number}"
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            return output_dir

        number += 1


def save_json(data, output_dir, filename):
    path = os.path.join(
        output_dir,
        filename
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved: {path}")


def save_results(raw_works, clean_works, yearly_stats):
    output_dir = create_output_dir()

    save_json(
        raw_works,
        output_dir,
        "raw_works.json"
    )

    save_json(
        clean_works,
        output_dir,
        "clean_works.json"
    )

    save_json(
        yearly_stats,
        output_dir,
        "yearly_stats.json"
    )

    print(f"\nResults saved to: {output_dir}")
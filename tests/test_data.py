import unittest

from prepare_data import CROPS_PER_SOURCE, make_quadruplets, split_source_ids


class DataSplitTests(unittest.TestCase):
    def test_sources_are_disjoint_and_cover_official_archives(self):
        splits = split_source_ids(42)
        self.assertEqual([len(ids) for ids in splits.values()], [700, 100, 100])
        combined = [source for sources in splits.values() for source in sources]
        self.assertEqual(sorted(combined), list(range(1, 901)))
        self.assertEqual(splits["test"], list(range(801, 901)))
        self.assertEqual(splits, split_source_ids(42))
        self.assertNotEqual(splits["validation"], split_source_ids(43)["validation"])

    def test_quadruplets_cover_all_crops_and_use_four_distinct_sources(self):
        sources = list(range(1, 13))
        records = make_quadruplets(sources, "train", 80, 42)
        self.assertEqual(len(records), 80)
        combinations = [tuple(record["images"]) for record in records]
        self.assertEqual(len(set(combinations)), 80)
        self.assertEqual(records, make_quadruplets(sources, "train", 80, 42))
        used_images = set()
        for record in records:
            self.assertEqual(len(record["images"]), 4)
            ids = [image.rsplit("/", 1)[1].split("_")[0] for image in record["images"]]
            self.assertEqual(len(set(ids)), 4)
            used_images.update(record["images"])
        self.assertEqual(
            used_images,
            {f"images/train/{source:04d}_{index:02d}.png"
             for source in sources for index in range(CROPS_PER_SOURCE)},
        )


if __name__ == "__main__":
    unittest.main()

import abc

import pandas as pd
import matplotlib.pyplot as plt
import argparse


class RegionFileProcessor(abc.ABC):
    def __init__(self, input_path: str) -> None:
        self.input_path = input_path
        self.df = pd.read_csv(
            self.input_path,
            sep='\t',
            header=None,
            names=['start', 'end']
        )

    @abc.abstractmethod
    def process(self) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def build_plot(self, df: pd.DataFrame) -> None:
        pass


class OverlappingRegionsStackUpProcessor(RegionFileProcessor):
    def process(self) -> pd.DataFrame:
        rows = []
        assigned = []

        sorted_idx = self.df["start"].sort_values().index

        for idx in sorted_idx:
            start = self.df.at[idx, "start"]
            end = self.df.at[idx, "end"]
            for row_num, last_end in enumerate(rows):
                if start > last_end:
                    assigned.append((idx, row_num + 1))
                    rows[row_num] = end
                    break
            else:
                rows.append(end)
                assigned.append((idx, len(rows)))

        assigned_rows = [0] * len(self.df)
        for idx, row in assigned:
            assigned_rows[idx] = row

        self.df.insert(0, "row", assigned_rows)

        output_path = self.input_path.replace(".txt", "_stacked.csv")
        self.df.to_csv(
            output_path,
            sep="\t",
            header=False,
            index=False,
            lineterminator="\r\n",
            encoding="utf-8",
        )
        return self.df

    def build_plot(self, df: pd.DataFrame) -> None:
        img_path = self.input_path.replace(".txt", "_stacked.png")
        plt.figure(figsize=(12, 4))
        for _, row in df.iterrows():
            plt.hlines(row["row"], row["start"], row["end"], linewidth=6)
        plt.xlabel("Position")
        plt.ylabel("Row")
        plt.title("Stacked Regions")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()


class OverlappingRegionsBySegmentsProcessor(RegionFileProcessor):
    def process(self) -> pd.DataFrame:
        boundaries = sorted(set(self.df["start"]).union(self.df["end"] + 1))
        segments = []
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1] - 1
            count = ((self.df["start"] <= seg_end) & (self.df["end"] >= seg_start)).sum()
            segments.append((count, seg_start, seg_end))

        seg_df = pd.DataFrame(segments, columns=["count", "start", "end"])
        output_path = self.input_path.replace(".txt", "_histogram.csv")
        seg_df.to_csv(
            output_path, sep="\t", header=False, index=False, lineterminator="\r\n"
        )
        return seg_df

    def build_plot(self, df: pd.DataFrame) -> None:
        img_path = self.input_path.replace(".txt", "_histogram.png")
        plt.figure(figsize=(12, 4))
        max_count = df["count"].max()
        for _, row in df.iterrows():
            alpha = 0.5 + 0.5 * (row["count"] / max_count) if max_count > 0 else 0.8
            alpha = min(max(alpha, 0), 1)
            plt.barh(
                0,
                row["end"] - row["start"] + 1,
                left=row["start"],
                height=0.8,
                color="tab:blue",
                alpha=alpha,
                edgecolor="black",
            )
        plt.yticks([])
        plt.xlabel("Position")
        plt.title("Coverage Depth Histogram")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stack overlapping regions from input file.')
    parser.add_argument('input_file', help='Path to the input file')
    args = parser.parse_args()

    processor = OverlappingRegionsStackUpProcessor(args.input_file)
    df = processor.process()
    processor.build_plot(df)

    processor = OverlappingRegionsBySegmentsProcessor(args.input_file)
    df = processor.process()
    processor.build_plot(df)

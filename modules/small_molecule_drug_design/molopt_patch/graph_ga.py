from molopt.graph_ga import GraphGA


class PatchGraphGA(GraphGA):
    def load_smiles_from_file(self, file_path):
        # read csv file
        import pandas as pd
        df = pd.read_csv(file_path)
        return df['smiles'].tolist()
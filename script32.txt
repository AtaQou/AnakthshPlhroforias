
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from colbert.data import Queries


if __name__ == '__main__':
   with Run().context(RunConfig(nranks=1, experiment="colbert_experiment")):
       config = ColBERTConfig(
           root="experiments"  # Διαδρομή για τα αποτελέσματα
       )


       searcher = Searcher(index="colbert_index", config=config)


       queries = Queries("queries.tsv")


       ranking = searcher.search_all(queries, k=100)
       ranking.save("colbert_index.ranking.tsv")
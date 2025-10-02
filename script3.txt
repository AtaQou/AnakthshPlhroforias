from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer


if __name__ == '__main__':
   with Run().context(RunConfig(nranks=1, experiment="colbert_experiment")):
       config = ColBERTConfig(
           nbits=2,
           root="experiments",  # Ο φάκελος όπου θα αποθηκευτούν τα αποτελέσματα του indexing
       )


       # Διαδρομή προς το φάκελο με το checkpoint
       checkpoint_path = "colbertv2.0"


       indexer = Indexer(checkpoint=checkpoint_path, config=config)
       indexer.index(name="colbert_index", collection="collection.tsv",overwrite=True )

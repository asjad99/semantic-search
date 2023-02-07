from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover, Tokenizer, CountVectorizer, IDF
import json
import nltk
from pyspark.mllib.linalg import Vectors
#from nltk.corpus import stopwords

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: <current-file-name>.py <in-file> <out-file> <out-model-file>", file=sys.stderr)
        exit(-1)

    spark = SparkSession\
        .builder\
        .appName("FeaturesExtraction")\
        .getOrCreate()

    in_path = sys.argv[1]
    print("---------- Stage 1: Reading files ---------- ")
    lines = spark.read.text(in_path).rdd.map(lambda r: json.loads(r[0].encode("utf-8"))["text"].lower())

    print(" ----------  Stage 2: Making Dataframe ----------  ")
    df = lines.map(lambda x: (nltk.word_tokenize(x), )).toDF(['words'])
    
    
    print(" ---------- Stage 3: Stopwords Removal ---------- ")
    
    # Filtering Documents, Filter includes removing stopwords and words with freq as 1.
    fl = StopWordsRemover(inputCol="words", outputCol="filtered")
    #fl.setStopWords(fl.getStopWords() + stopwords.words('english'))
    df = fl.transform(df).select("filtered")
    df.show(truncate = False)

    print("----------  Stage 4: CountVectorizer ---------- ")
    #  Extracting Features based on filtered words.
    cv = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")
    model = cv.fit(df)
    df = model.transform(df).select("rawFeatures")

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(df)
    rescaledData = idfModel.transform(df).select("features")
    rescaledData.show(truncate = False)
    rescaledData.printSchema()
    from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, RowMatrix
    from pyspark.mllib.linalg import Vectors

    mat = RowMatrix(df.rdd.map(lambda v: Vectors.dense(v.rawFeatures.toArray()) ))

    svd = mat.computeSVD(2, computeU=True)
    U = svd.U       # The U factor is a RowMatrix.
    s = svd.s       # The singular values are stored in a local dense vector.
    V = svd.V       # The V factor is a local dense matrix.

    #-----------------------------------------------------------------------
    #LSI 
    #-----------------------------------------------------------------------

    from gensim import corpora, models, similarities
    dictionary = corpora.Dictionary()
    dictionary.token2id = {unicode(item): i for i,item in enumerate(model.vocabulary)}

    lsi = models.LsiModel([], id2word=dictionary, num_topics=2)
    import numpy as np
    lsi.projection.u = np.array(U.rows.collect())
    lsi.projection.s = s

    print("---------- Stage 5: Generating Corpus ---------- ")
    # Generating Corpus.
    def f(row):
        jvs = (zip(row.rawFeatures.indices, row.rawFeatures.values) if hasattr(row.rawFeatures, "indices") else enumerate(row.rawFeatures.toArray()))
        return [(j, v) for j, v in jvs if v]

    corpus = df.rdd.map(f).collect()
    print(dictionary.token2id)
    print(corpus)
    print(lsi.projection.u)
    print(lsi.projection.s)
    print(lsi.print_topics(10))


    corpus_lsi = lsi[corpus]
    print("corpus_lsi {}".format(corpus_lsi))
    index = similarities.MatrixSimilarity(corpus_lsi)
    
    #---------------------------------------------------
    #query = "Human computer interaction"
    #query_bow = dictionary.doc2bow(query.lower().split())
    #print("Bag of words {}".format(query_bow))
    #query_lsi = lsi[query_bow] 
    
    spark.stop()

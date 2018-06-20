library(stringr)
library(text2vec)
library(tokenizers)
library(data.table)


# text file reader
FILE_PATH <- "Z:/Avito/matt/"


for (i in c(1, 2)) {
  # load data
  file_name <- sprintf('%s/text/text_%d.txt', FILE_PATH, i)
  text <- fread(file_name, sep='\n', header=F, encoding="UTF-8")[[1]]

  # tokenize
  it <- itoken(text, tokenizer = space_tokenizer)
  
  # Create vocabular, terms will be unigrams
  vocab <- create_vocabulary(it) %>% prune_vocabulary(term_count_min=1L)
  vectorizer <- vocab_vectorizer(vocab)
  
  # use window of 5 for context words
  tcm <- create_tcm(it, vectorizer, skip_grams_window=5L)
  
  # create vectors of length 100, 50
  for (n in c(300)) {
    
    # create glove vectors
    glove <- GlobalVectors$new(word_vectors_size=n, vocabulary=vocab, x_max=10)
    wv_matrix <- glove$fit_transform(tcm, n_iter=50, convergence_tol=1e-6)
    wv_matrix <- round(wv_matrix, 5)

    # convert to data.table
    wv_table <- data.table(wv_matrix)
    rownames(wv_table) <- rownames(wv_matrix)
    
    # save data
    save_name <- sprintf('%s/embeddings/glove_avito_%dd_%d.txt', FILE_PATH, n, i)
    fwrite(wv_table, save_name, sep=' ', row.names=T, col.names=F)  
  }
}



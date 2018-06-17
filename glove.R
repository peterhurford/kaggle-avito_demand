options(stringsAsFactors = FALSE,
        menu.graphics = FALSE,
        repos = structure(c(CRAN = "https://ftp.ussg.iu.edu/CRAN/")))

if (Sys.getenv("R_LIBS_USER") %in% .libPaths()) {
	main_lib <- normalizePath(Sys.getenv("R_LIBS_USER"))
} else {
	main_lib <- .libPaths()[[1]]
}

if (!require(stringr)) { install.packages("stringr") }; library(stringr)
if (!require(text2vec)) { install.packages("text2vec") }; library(text2vec)
if (!require(tokenizers)) { install.packages("tokenizers") }; library(tokenizers)
if (!require(data.table)) { install.packages("data.table") }; library(data.table)


# text file reader
for (i in c(1, 2, 0)) {
  message(i)

  # load data
  file_name <- sprintf("cache/text_%d.txt", i)
  text <- fread(file_name, sep = "\n", header = FALSE, encoding = "UTF-8")[[1]]

  # tokenize
  it <- itoken(text, tokenizer = space_tokenizer)
  
  # Create vocabular, terms will be unigrams
  vocab <- create_vocabulary(it) %>% prune_vocabulary(term_count_min = 1L)
  vectorizer <- vocab_vectorizer(vocab)
  
  # use window of 5 for context words
  tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
  
  # create vectors of length 100, 50
  for (n in c(100, 50)) {
    message(paste0("...", as.character(n)))
    # create glove vectors
    glove <- GlobalVectors$new(word_vectors_size = n, vocabulary = vocab, x_max = 10)
    wv_matrix <- glove$fit_transform(tcm, n_iter = 50, convergence_tol = 1e-6)
    
    # convert to data.table
    wv_table <- data.table(wv_matrix)
    rownames(wv_table) <- rownames(wv_matrix)
    
    # save data
    save_name <- sprintf("cache/glove_avito_300d_%d.txt", i)
    fwrite(wv_table, save_name, sep=" ", row.names = TRUE, col.names = FALSE)  
  }
}


## notes
# - toxic_basic_glove_200d, training error
# - toxic.sarcasm.glove200, training error
# - toxic.200, training error ')}}

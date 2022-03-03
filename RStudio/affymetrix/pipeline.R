#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("affy")
#aBiocManager::install("preprocessCore", configure.args="--disable-threading")
library(affy)
library(stringr)

accession_list <- c("GSE10846", "GSE31312") # DLBCL
cel_filelist <- NULL
duplicate_check <- NULL

for (geo_no in accession_list){
  DIR <- "/pytorch/docker/MainData" # data directory
  DIR <- paste(DIR, geo_no, sep="/")
  files <- list.celfiles(DIR)
  print(length(files))
  duplicate_check <- append(duplicate_check, files)
  files <- str_c(DIR, files, sep="/")
  cel_filelist <- append(cel_filelist, files)
  
  SAVE_DIR <- "/pytorch/docker/RStudio/affymetrix/" # save directory

  cel_set <- ReadAffy(filenames=files, compress = TRUE)
  
  #file_name <- paste(SAVE_DIR, geo_no, "_mas5_expression.txt", sep="")
  #mas5_eset <- mas5(cel_set)
  #write.exprs(mas5_eset, file=file_name)
  
  file_name <- paste(SAVE_DIR, geo_no, "_rma_expression.txt", sep="")
  rma_eset <- rma(cel_set)
  write.exprs(rma_eset, file=file_name)
  
}
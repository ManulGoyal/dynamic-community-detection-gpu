install.packages("igraph")
library(igraph)
setwd(".../myfolder")
#simplify all input graphs (deletion and addition files)
file.names <- dir(".../myfolder", pattern ="add")
for(i in 1:n){#n is the number of csv files in the directory
  g <- read_graph(file.names[i],"edgelist")
  g<- simplify(g, remove.loops=FALSE)
  g <- simplify(g, remove.multiple=FALSE)
  g <- as.undirected(g)
  write_graph(g, file.names[i],"edgelist")
}

file.names <- dir(".../myfolder", pattern ="sub")
for(i in 1:n){
  g <- read_graph(file.names[i],"edgelist")
  g<- simplify(g, remove.loops=FALSE)
  g <- simplify(g, remove.multiple=FALSE)
  g <- as.undirected(g)
  write_graph(g, file.names[i],"edgelist")
}

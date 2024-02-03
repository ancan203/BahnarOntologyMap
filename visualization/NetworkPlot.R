library(htmltools)
library(networkD3)
library(htmlwidgets)
library(rmarkdown)
library(dplyr) # to make the joins easier
library(openxlsx)
install.packages("devtools")


nodes$config <- paste(nodes$name, nodes$node_name, sep=": ")

detach("package:networkD3", unload = TRUE)
install.packages("networkD3", dependencies=TRUE)
devtools::install_github("fraupflaume/networkD3", force = TRUE) 

src <- c(data$Keys)
target <- c(data$Representation)

node_name <- unique(c(data$Document))
networkData <- data.frame(src, target, stringsAsFactors = FALSE)
newID <- unique(c(target))



# create a data frame of the edges that uses id 0:9 instead of their names
edges <- networkData %>%
  left_join(nodes, by = c("src" = "name")) %>%
  select(-src) %>%
  rename(source = id) %>%
  left_join(nodes, by = c("target" = "name")) %>%
  select(-target) %>%
  rename(target = id)

edges$width <- 1




MyClickScript <- 
  '      d3.select(this).select("circle").transition()
.duration(750)
.attr("r", 30)'
# simple with default colours


fn <- forceNetwork(Links = edges, Nodes = nodes, 
             Source = "source",
             Target = "target",
             NodeID ="config",
             Group = "group",
             Value = "width",
             Nodesize = "size",
             opacity = 0.9,
             zoom = TRUE,
             fontFamily = "sans-serif",
             
             charge = -10)



fn



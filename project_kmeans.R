setwd("/Users/zahrahalim/desktop/INFO911/project")



#------------------KMEANS------------------------------

df<- train

# turn zeros into NA

df[(df==0)]<-NA

# remove the missing values 

df<- na.omit(df)

# scale the data to mean zero and sd 1

ds<-normalizeFeatures(df, method = "standardize")

# remove colomuns with zeros
col_z <- lapply(ds, function(col){length(which(col==0))/length(col)})
ds <- ds[, !(names(ds) %in% names(col_z[lapply(col_z, function(x) x) > 0.9]))]

# remove the class as kmeans only works with variables with numeric data
ds$class = NULL

# compute kmeans

result<- kmeans(ds, 2)

attributes(result)

str(result)

# compare with original data class

table(train$class, result$cluster)

# visalizing distance matrix
#red means large dissimilarities and teal suggest fairly similar

distance<- get_dist(ds)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))           

# illustration of cluster using principal component analysis(PCA)

fviz_cluster(result, data = ds)

# using different center values

result3 <- kmeans(ds, centers = 3, nstart = 25)
result4 <- kmeans(ds, centers = 4, nstart = 25)
result5 <- kmeans(ds, centers = 5, nstart = 25)

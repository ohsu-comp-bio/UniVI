
A single object matching ‘calcAlignment’ was found
It was found in the following places
  package:rliger
  namespace:rliger
with value

function (object, k = NULL, rand.seed = 1, cells.use = NULL, 
    cells.comp = NULL, clusters.use = NULL, by.cell = FALSE, 
    by.dataset = FALSE) 
{
    if (is.null(cells.use)) {
        cells.use <- rownames(object@H.norm)
    }
    if (!is.null(clusters.use)) {
        cells.use <- names(object@clusters)[which(object@clusters %in% 
            clusters.use)]
    }
    if (!is.null(cells.comp)) {
        nmf_factors <- object@H.norm[c(cells.use, cells.comp), 
            ]
        num_cells <- length(c(cells.use, cells.comp))
        func_H <- list(cells1 = nmf_factors[cells.use, ], cells2 = nmf_factors[cells.comp, 
            ])
        message("Using designated sets cells.use and cells.comp as subsets to compare")
    }
    else {
        nmf_factors <- object@H.norm[cells.use, ]
        num_cells <- length(cells.use)
        func_H <- lapply(seq_along(object@H), function(x) {
            cells.overlap <- intersect(cells.use, rownames(object@H[[x]]))
            if (length(cells.overlap) > 0) {
                object@H[[x]][cells.overlap, ]
            }
            else {
                warning(paste0("Selected subset eliminates dataset ", 
                  names(object@H)[x]), immediate. = TRUE)
                return(NULL)
            }
        })
        func_H <- func_H[!sapply(func_H, is.null)]
    }
    num_factors <- ncol(object@H.norm)
    N <- length(func_H)
    if (N == 1) {
        warning("Alignment null for single dataset", immediate. = TRUE)
    }
    set.seed(rand.seed)
    min_cells <- min(sapply(func_H, function(x) {
        nrow(x)
    }))
    sampled_cells <- unlist(lapply(1:N, function(x) {
        sample(rownames(func_H[[x]]), min_cells)
    }))
    max_k <- length(sampled_cells) - 1
    if (is.null(k)) {
        k <- min(max(floor(0.01 * num_cells), 10), max_k)
    }
    else if (k > max_k) {
        stop(paste0("Please select k <=", max_k))
    }
    knn_graph <- get.knn(nmf_factors[sampled_cells, 1:num_factors], 
        k)
    if (!is.null(cells.comp)) {
        dataset <- unlist(sapply(1:N, function(x) {
            rep(paste0("group", x), nrow(func_H[[x]]))
        }))
    }
    else {
        dataset <- unlist(sapply(1:N, function(x) {
            rep(names(object@H)[x], nrow(func_H[[x]]))
        }))
    }
    names(dataset) <- rownames(nmf_factors)
    dataset <- dataset[sampled_cells]
    num_sampled <- N * min_cells
    num_same_dataset <- rep(k, num_sampled)
    alignment_per_cell <- c()
    for (i in 1:num_sampled) {
        inds <- knn_graph$nn.index[i, ]
        num_same_dataset[i] <- sum(dataset[inds] == dataset[i])
        alignment_per_cell[i] <- 1 - (num_same_dataset[i] - (k/N))/(k - 
            k/N)
    }
    if (by.dataset) {
        alignments <- c()
        for (i in 1:N) {
            start <- 1 + (i - 1) * min_cells
            end <- i * min_cells
            alignment <- mean(alignment_per_cell[start:end])
            alignments <- c(alignments, alignment)
        }
        return(alignments)
    }
    else if (by.cell) {
        names(alignment_per_cell) <- sampled_cells
        return(alignment_per_cell)
    }
    return(mean(alignment_per_cell))
}
<bytecode: 0x55e1d0dc5628>
<environment: namespace:rliger>

# Generate standard visualization and analyses from data

dat <- read.csv("outfile.csv")
dat$shuffled <- ifelse(dat$shuffled == "True", "Shuffled", "Not Shuffled")
dat$blocks <- factor(dat$blocks)
dat$gamma.init.rep <- factor(dat$gamma.init.rep)
dat.mix <- dat[dat$shuffled == "Shuffled",]
dat.hmm <- dat[dat$shuffled == "Not Shuffled",]

# Setup plotting library
require(lattice)
trellis.par.set(superpose.symbol = list(col = c("red", "blue")))


pdf("all_run_time_num_data_by_block.pdf")
with(dat, xyplot(run.time ~ num.data | blocks + shuffled))
dev.off()

pdf("all_err_mean_max_num_data_by_block.pdf")
with(dat, xyplot(err.mean.max ~ num.data | blocks + shuffled))
dev.off()

pdf("all_err_mean_mean_num_data_by_block.pdf")
with(dat, xyplot(err.mean.mean ~ num.data | blocks, groups = shuffled,
                 scales = list(alternating = FALSE),
                 strip = strip.custom(strip.names = c(TRUE, TRUE)),
                 auto.key = list(space = "bottom"),
                 layout = c(NA,1),
                 main = "EM Performance with Blocking",
                 xlab = "number of data",
                 ylab = "mean absolute error in location parameter"))
dev.off()

pdf("all_log_likelihood_num_data_by_block.pdf")
with(dat, xyplot(log.likelihood ~ num.data | blocks + shuffled))
dev.off()

pdf("all_reps_num_data_by_block.pdf")
with(dat, xyplot(reps ~ num.data | blocks + shuffled))
dev.off()

pdf("hmm_run_time_block_by_num_data.pdf")
with(dat.hmm, xyplot(run.time ~ blocks | cut(num.data,4)))
dev.off()

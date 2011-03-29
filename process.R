# Generate standard visualization and analyses from data

dat <- read.csv("outfile.csv")
dat$blocks <- factor(dat$blocks)
dat$gamma.init.rep <- factor(dat$gamma.init.rep)
dat.mix <- dat[dat$shuffled == "True",]
dat.hmm <- dat[dat$shuffled == "False",]

require(lattice)

pdf("all_run_time_num_data_by_block.pdf")
with(dat, xyplot(run.time ~ num.data | blocks + shuffled))
dev.off()

pdf("all_reps_num_data_by_block.pdf")
with(dat, xyplot(reps ~ num.data | blocks + shuffled))
dev.off()

pdf("hmm_run_time_block_by_num_data.pdf")
with(dat.hmm, xyplot(run.time ~ blocks | cut(num.data,4)))
dev.off()

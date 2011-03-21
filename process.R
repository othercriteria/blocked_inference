# Generate standard visualization and analyses from data

dat <- read.csv("outfile.csv")
dat.mix <- dat[dat$model.type == "Mixture",]
dat.hmm <- dat[dat$model.type == "HMM",]

require(lattice)
pdf("run_time_num_data_by_block.pdf")
with(dat.hmm, xyplot(run.time ~ num.data | factor(blocks)))
dev.off()

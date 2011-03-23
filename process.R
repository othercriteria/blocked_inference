# Generate standard visualization and analyses from data

dat <- read.csv("outfile.csv")
dat.mix <- dat[dat$model.type == "Mixture",]
dat.hmm <- dat[dat$model.type == "HMM",]

require(lattice)

pdf("all_run_time_num_data_by_block.pdf")
with(dat, xyplot(run.time ~ num.data | factor(blocks) + model.type))
dev.off()

pdf("hmm_run_time_block_by_num_data.pdf")
with(dat.hmm, xyplot(run.time ~ blocks | cut(num.data,4)))
dev.off()

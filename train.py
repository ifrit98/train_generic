from train_generic import training_run, inside_docker, FLAGS
runs_dir = "/training/data/runs" if inside_docker() else "/array1/data/front_row_data/runs"
training_run(FLAGS, runs_dir=runs_dir)
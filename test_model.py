from deepforest import main

print("Loading DeepForest...")

model = main.deepforest()

model.load_model(
    model_name="weecology/deepforest-tree",
    revision="main"
)

print("DeepForest loaded successfully!")
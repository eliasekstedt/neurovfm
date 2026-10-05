
# notes
* how can information be preserved for classifier when .mean() is done between all patches?
    * realized that mean pooling will likely lead to less focused heatmaps. either do simple logit based method or switch to attention pooling.
* getting explanation methods working is a goal in and of itself ... mostly..

# do
* decide if your deeplift implementation actually worked
    * generate attr without up-process, try to see what computations if any can can result in more clear heatmaps.
* background removal really removes a lot, and im not sure it is necessary. how about try generating new features without removing bg.
* look at how others have implemented DeepLift with 3d vision transformers
* read about deeplift, check their code, use copilot to understand


# done
* build DeepLift framework that includes up/down processing
# Model Info

The original RITnet model is labeled as `densenet_og.py` because of the Dense Blocks found in RITnet. 

Our tested models incorporate Inverted Residual Blocks (IRB) found in MobileNet. We replaced a varying number of Dense Encoding Blocks with IRBs.
* `mobilenet_v1`: 2nd encoding dense block replaced with IRB
* `mobilenet_v2`: 1st and 2nd encoding dense blocks replaced with IRBs
* `mobilenet_v3`: 1st-3rd encoding dense blocks replaced with IRBs
* `mobilenet_v4`: 1st-4th encoding dense blocks replaced with IRBs
* `mobilenet_v5`: 1st-5th (all) encoding dense blocks replaced with IRBs
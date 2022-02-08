# DigitalHumanitiesProject
Source code for digital humanities project conducted at UIUC.


## Setup

To ensure consistency across platforms, the source code is run through a Docker container (an image can be built with the included Dockerfile).

The following command starts the environment, when run from within the Docker container: 

```bash
> conda activate dhp
```

The same process can alternatively be performed via IDEs: a Docker configuration can be set up to optimize usage. 

## Running Source Code

Source code can be run manually through Python or make.
```bash
# analyzes a single image and generates outputs
> python3 driver.py in.png out.png 

# analyzes all images in the images directory and generates an aggregated JSON.
> make exec 

# takes all currently-analyzed images and generates an aggregated JSON.
> make aggregator 
```

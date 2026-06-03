#!/bin/bash
gmt begin $1 png
    gmt basemap -Ra143.grd -JM10c -Baf
    gmt plot $1 -W1p,red
gmt end show

- removed invlaid latitute and longitude because they made the largest distance between two coordinate points being 600km, I realised this when I tried to find the max distance to normalise the similarity function 
- number of days in months are always different. but to still have them cyclical so that 31.04 is close to 01.05, I used 31 to make it cyclical 
- i took out the day of the month parameter from the temporal distance function as its first of all hard to turn it into cyclical and its also negligent to the patterns we are trying to find. for us, more important is the weekday, the month (season) and the time of the day
- when reviewing a pull request on the visualisations, i realised that there is a typo, where there is both CRIM SEXUAL ASSAULT and CRIMINAL SEXUAL ASSAULT for primary crime type. i tried finding info on this on by researching the Illinois Uniform Crime Reporting (IUCR) Codes. However, there was no info on this. It seems, it is a typo. I therefore renamed all occurences of CRIM SEXUAL ASSAULT to be CRIMINAL SEXUAL ASSAULT
- Load eagerly at startup, or lazily on first query per type?

Eager: Loop through data/precomputed/knn/*.npz at app startup, load all into a dict in memory. First query for any type is fast; startup is slower; memory usage is the sum of all artifacts (probably <100 MB total based on what you saw).
Lazy: Load each type's .npz only when first queried, cache it. Startup is instant; first query per type pays the load cost; long-running app eventually has everything in memory anyway.

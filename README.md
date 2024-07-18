# Final Assignment: Robust Journey Planning

PREVIEW

*Executive summary:* Build a robust SBB journey planner, and make a short video presentation of it, to be done as a *team project*.
* A short (7min max) video presentation of your work is due before *May 28th, 23:59 CEST*
* Code is due before *May 31, 23:59 CEST*.

----
## HOW-TO

In the beginning of each notebook, select a region name and id (be careful to select the same for each notebook!). From there, start by running Model_transport.ipynb to create the dataframes for the different parts. Then, to obtain the predictions, run Predictions_delay.ipynb. To finish, in the notebook Route_Planning.py, run it in its totallity to obtain, at the end, the final visualization. To choose a source, a destination and an arrival time for your journey, go to the end of the notebook, and in the last cell, modify as wanted, as long as the stops are in the connected graph.

_Video_: https://drive.google.com/drive/folders/14AL_4jEZhNps8WG-QZU66KibWY3WzrEQ


----
## Problem Motivation

Imagine you are a regular user of the public transport system, and you are checking the operator's schedule to meet your friends for a class reunion.
The choices are:

1. You could leave in 10mins, and arrive with enough time to spare for gossips before the reunion starts.

2. You could leave now on a different route and arrive just in time for the reunion.

Undoubtedly, if this is the only information available, most of us will opt for option 1.

If we now tell you that option 1 carries a fifty percent chance of missing a connection and be late for the reunion. Whereas, option 2 is almost guaranteed to take you there on time. Would you still consider option 1?

Probably not. However, most public transport applications will insist on the first option. This is because they are programmed to plan routes that offer the shortest travel times, without considering the risk factors.

----
## Problem Description

In this final project you will build your own robust public transport route planner to improve on that. You will reuse the SBB dataset (See next section: [Dataset Description](#Dataset-Description)).

Given a desired arrival time, your route planner will compute the fastest route between departure and arrival stops within a provided confidence tolerance expressed as interquartiles.
For instance, "what route from A to B is the fastest at least Q% of the time if I want to arrive at B before instant T". Note that confidence is a measure of a route being feasible within the travel time computed by the algorithm.

The output of the algorithm is a list of routes between A and B and their confidence levels. The routes must be sorted from latest (fastest) to earliest (longest) departure time at A, they must all arrive at B before T with a confidence level greater than or equal to Q. Ideally, it should be possible to visualize the routes on a map with straight lines connecting all the stops traversed by the route.

In order to answer this question you will need to:

- Model the public transport infrastructure for your route planning algorithm using the data provided to you.
- Build a predictive model using the historical arrival/departure time data, and optionally other sources of data.
- Implement a robust route planning algorithm using this predictive model.
- Test and *validate* your results. Note that we will put a particular emphasis on the scientific validation of your method.
- Implement a simple Jupyter-based visualization to demonstrate your method, using Jupyter widgets such as [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_guide.html).

Solving this problem accurately can be difficult. You are allowed a few *simplifying assumptions*:

- We only consider journeys at reasonable hours of the day, and on a typical business day, and assuming a recent schedule.
- We allow short (total max 500m "As the Crows Flies") walking distances for transfers between two stops, and assume a walking speed of 50m/1min on a straight line, regardless of obstacles, human-built or natural, such as building, highways, rivers, or lakes.
- We will only consider journeys that start and end on known station coordinates (train station, bus stops, etc.), never from a random location. However, walking from the departure stop to a nearby stop is allowed.
- We only consider stops in an area that will be specified at a later time.
- The boundaries of the area are approximative, if needed stops may be reached via transfers through other stops outside the 15km area.
- There is no penalty for assuming that delays or travel times on the public transport network of *two different lines* are uncorrelated with one another, *however* try to capture dependenes if possible, such as dependencies of delays on a same route;
 e.g. if a train is late at a stop, it is expected to be late at subsequent stops.
- Once a route is computed, a traveller is expected to follow the planned routes to the end, or until it fails (i.e. miss a connection).
  You *do not* need to address the case where travellers are able to defer their decisions and adapt their journey "en route", as more information becomes available. This would require us to consider all alternative routes (contingency plans) in the computation of the uncertainty levels, which is more difficult to implement.
- The planner will not need to mitigate the traveller's inconvenience if a plan fails. Two routes with identical travel times under the uncertainty tolerance are equivalent, even if the outcome of failing one route is much worse for the traveller than failing the other route, such as being stranded overnight on one route and not the other.
- All other things being equal, we will prefer routes with the minimum walking distance and minimum number of transfers.
- You do not need to optimize the computation time of your method, as long as the run-time is reasonable.
- When computing a path you may pick the timetable of a recent week and assume that it is unchanged.

Upon request, and with clear instructions from you, we can help prepare the data in a form that is easier for you to process (within the limits of our ability, and time availability). In which case the data will be accessible to all.

----
## Project Submission Checklist

1. Project (code) deadline is May 30th, 23:59 CEST.

1. The final assignment is a *team* project, remember to update your group member list if needed.

1. The assignment is a python code and/or well-annotated notebook; report-like.

1. All projects must be submitted to gitlab, as a group project (under your group name).

1. Project must contain final in the name.

1. Provide instructions on how to test your project in the *HOW TO* section of the README.md file. Include a link to your video presentation.

1. Project sizes, including history, must not exceed 100Mb. Use HDFS for your larger data sets.

----
## Video Presentations Checklist

1. Video presentation of your work is due before May 28th, 23:59 CEST.

1. If possible, try to give some air time to each member of the team.

1. Use Zoom (or other tools) to record your group video.

1. Save the video as an mp4 file.

1. Upload your video to moodle under Final assignment - video presentation.

1. Include the link to the video in the *HOW TO* section, at the top of the README.md file of your final assignment

Please, *DO NOT* save the video in your git project, *DO NOT* send a video embedded in a PowerPoint presentations, and *DO NOT use any format other than mp4 videos. We must be able to stream the videos in our web browsers.

---- 
## Grading Method

After reviewing your videos, we will invite each group for a 8 mins Q&A. Before the Q&A, we may validate your method on a list of pre-selected departure arrival points, and times of day.

For the oral defense, we will organize short Q&A discussions of 8 minutes per group. These discussions will be scheduled on the week of *June 3th* - actual day and times to be discussed on a case by case basis.

Think of yourselves as a startup trying to sell your solution to the board of a public transport
company. Your video is your elevator pitch. It must be short and convincing. In it you describe the viability
of the following aspects:

1. Method used to model the public transport network
2. Method used to create the predictive models
3. Route planning algorithm
4. Validation method

Your grades will be based on the code, videos and Q&A, taking into account:

1. Clarity and conciseness of the video presentation, code and Q&A
2. Team work, formulation and decomposition of the problem into smaller tasks between team members
3. Originality of the solution design, analytics, and presentation
4. Functional quality of the implementation (does it work?)
5. Explanation of the pro's and con's / shortcomings of the proposed solution

----
## Dataset Description

For this project we will use the data published on the [Open Data Platform Mobility Switzerland](<https://opentransportdata.swiss>).
For grading we will use recent SBB data limited around a city area of our choice (hold out data) ~the Zurich area, focusing only on stops within 15km of the Zurich main train station~. We recommend that you develop a solution around Lausanne area in the same region as Assignment 2 and make sure that the solution will be able to work in a different or slightly larger area. We will select the area using a different region from the com490.geo_shapes table. The region objectid from the table should therefore be a well documented parameter of your solution (in configuration file or variable in the notebook). 

We will also provide you with a simulated realtime feed of Istdaten data.

For your convenience, we have defined the table for each of the data set described below in the Hive database com490. You can list the tables with the command SHOW TABLES IN com490;.  *DO NOT* drop the database, *DO NOT* drop the tables.

#### Actual data

Students should already be familiar with the [istdaten](https://opentransportdata.swiss/de/dataset/istdaten)  data.

The 2020 to 2024 data is available as a Hive table in partitioned ORC format on our HDFS system, under the /data/sbb/{orc|parquet}/istdaten folder.

See assignments and exercises of earlier weeks for more information about this data, and methods to access it.

As a reminder, we provide the relevant column descriptions below.
The full description of the data is available in the opentransportdata.swiss data [istdaten cookbooks](https://opentransportdata.swiss/en/cookbook/actual-data/).

* *istdaten*
    - BETRIEBSTAG: date of the trip
    - FAHRT_BEZEICHNER: identifies the trip
    - BETREIBER_ABK, BETREIBER_NAME: operator (name will contain the full name, e.g. Schweizerische Bundesbahnen for SBB)
    - PRODUCT_ID: type of transport, e.g. train, bus
    - LINIEN_ID: for trains, this is the train number
    - LINIEN_TEXT,VERKEHRSMITTEL_TEXT: for trains, the service type (IC, IR, RE, etc.)
    - ZUSATZFAHRT_TF: boolean, true if this is an additional trip (not part of the regular schedule)
    - FAELLT_AUS_TF: boolean, true if this trip failed (cancelled or not completed)
    - HALTESTELLEN_NAME: name of the stop
    - ANKUNFTSZEIT: arrival time at the stop according to schedule
    - AN_PROGNOSE: actual arrival time (see AN_PROGNOSE_STATUS)
    - AN_PROGNOSE_STATUS: method used to measure AN_PROGNOSE, the time of arrival.
    - ABFAHRTSZEIT: departure time at the stop according to schedule
    - AB_PROGNOSE: actual departure time (see AN_PROGNOSE_STATUS)
    - AB_PROGNOSE_STATUS: method used to measure  AB_PROGNOSE, the time of departure.
    - DURCHFAHRT_TF: boolean, true if the transport does not stop there

Each line of the file represents a stop and contains arrival and departure times. When the stop is the start or end of a journey, the corresponding columns will be empty (ANKUNFTSZEIT/ABFAHRTSZEIT).
In some cases, the actual times were not measured so the AN_PROGNOSE_STATUS/AB_PROGNOSE_STATUS will be empty or set to PROGNOSE and AN_PROGNOSE/AB_PROGNOSE will be empty.

#### Timetable data

Timetable data are available from opentransportdata.swiss's [timetable](https://opentransportdata.swiss/en/cookbook/gtfs/) data set.

The timetables are updated weekly. It is ok to assume that the weekly changes are small, and a timetable for
a given week is thus the same for the full year - use the schedule of the most recent week for the day of the trip. However, note that public transport services may run at different times or not at all depending on the day of the week.

The full description of the GTFS format is available in the opentransportdata.swiss data [timetable cookbooks](https://opentransportdata.swiss/en/cookbook/gtfs/).

We provide a summary description of the data below:

* *stops*:

    - STOP_ID: unique identifier (PK) of the stop
    - STOP_NAME: long name of the stop
    - STOP_LAT: stop latitude (WGS84)
    - STOP_LON: stop longitude
    - LOCATION_TYPE:
    - PARENT_STATION: if the stop is one of many collocated at a same location, such as platforms at a train station

* *stop_times*:

    - TRIP_ID: identifier (FK) of the trip, unique for the day - e.g. 1.TA.1-100-j19-1.1.H
    - ARRIVAL_TIME: scheduled (local) time of arrival at the stop (same as DEPARTURE_TIME if this is the start of the journey)
    - DEPARTURE_TIME: scheduled (local) time of departure at the stop 
    - STOP_ID: stop (station) identifier (FK), from stops.txt
    - STOP_SEQUENCE: sequence number of the stop on this trip id, starting at 1.
    - PICKUP_TYPE:
    - DROP_OFF_TYPE:

* *trips*:

    - ROUTE_ID: identifier (FK) for the route. A route is a sequence of stops. It is time independent.
    - SERVICE_ID: identifier (FK) of a group of trips in the calendar, and for managing exceptions (e.g. holidays, etc).
    - TRIP_ID: is one instance (PK) of a vehicle journey on a given route - the same route can have many trips at regular intervals; a trip may skip some of the route stops.
    - TRIP_HEADSIGN: displayed to passengers, most of the time this is the (short) name of the last stop.
    - TRIP_SHORT_NAME: internal identifier for the trip_headsign (note TRIP_HEADSIGN and TRIP_SHORT_NAME are only unique for an agency)
    - DIRECTION_ID: if the route is bidirectional, this field indicates the direction of the trip on the route.
    
* *calendar*:

    - SERVICE_ID: identifier (PK) of a group of trips sharing a same calendar and calendar exception pattern.
    - MONDAY..SUNDAY: FALSE (0) or TRUE (1) for each day of the week, indicating occurence of the service on that day.
    - START_DATE: start date when weekly service id pattern is valid
    - END_DATE: end date after which weekly service id pattern is no longer valid
    
* *routes*:

    - ROUTE_ID: identifier for the route (PK)
    - AGENCY_ID: identifier of the operator (FK)
    - ROUTE_SHORT_NAME: the short name of the route, usually a line number
    - ROUTE_LONG_NAME: (empty)
    - ROUTE_DESC: Bus, Zub, Tram, etc.
    - ROUTE_TYPE:
    
*Note:* PK=Primary Key (unique), FK=Foreign Key (refers to a Primary Key in another table)

The other files are:

* calendar-dates.txt contains exceptions to the weekly patterns expressed in calendar.txt.
* agency.txt has the details of the operators
* transfers.txt contains the transfer times between stops or platforms.


#### Geo spatial data

You will find useful geospatial shapes in the Hive table com490.geo_shapes.

* com490.geo_shapes:
    * OBJECTID: numerical identifier of the shape
    * NAME: a human readable name of the shape
    * GEOMETRY: binary representation of the shape that can be used in ESRI UDF functions.

#### Misc data

Althought, not required for this final, you are of course free to use any other sources of data of your choice that might find helpful.

You may for instance download regions of openstreetmap [OSM](https://www.openstreetmap.org/#map=9/47.2839/8.1271&layers=TN),
which includes a public transport layer. If the planet OSM is too large for you,
you can find frequently updated exports of the [Swiss OSM region](https://planet.osm.ch/).

Others had some success using weather data to predict traffic delays.
If you want to give a try, web services such as [wunderground](https://www.wunderground.com/history/daily/ch/r%C3%BCmlang/LSZH/date/2022-1-1), can be a good
source of historical weather data. For your convenience we have made available to you a copy of the last three years of weather data reported at various airports in Swizerland. You can find the data on HDFS under /data/wunderground.

----
## Hints

Before you get started, we offer a few hints:

- Reserve some time to Google-up the state of the art before implementing. There is a substantial amount of work on this topic. Look for time-dependent, or time-varying networks, and stochastic route planning under uncertainty.
- You should already be acquainted with the data.
However, as you learn more about the state of the art, spend time to better understand your data.
Anticipate what can and cannot be done from what is available to you, and plan your design strategy accordingly. Do not hesitate to complete the proposed data sources with your own if necessary.
- Start small with a simple working solution and improve on it.
In a first version, assume that all trains and buses are always sharp on time.
Focus on creating a sane collaborative environment that you can use to develop and test your work in team as it evolves.
Next, work-out the risk-aware solution gradually - start with a simple predictive model and improve it. In addition you can test your algorithm on selected pairs of stops before generalizing to the full public transport network under consideration.

----
## FAQ

This section will be updated with the Frequently Asked Questions during the course of this project. Please stay tuned.

##### 1 - Q: Do we need to take into account walking times at the connections?
* *A*: Yes, but since we do not have the details of the platforms at each location, we can use a universal formula to come up with a reasonable walking time.
We must also allow time for transfers between different modes of transports, such as from bus to tramways.
You can use the transfer time information available from transfers.txt from the [timetables](#timetable-data).
Otherwise, we assume that 2min mininum are required for transfers within a same location
(i.e. same lat,lon coordinates), to which you add 1min per 50m walking time
to connect two stops that are at most 500m appart, on a straight line distance between their two lat,lon. 

##### 2 - Q: Can we assume statistical independence between the observed delays?
* *A: Yes, see simplifying assumptions in **Problem Description*.
You will incur no penalty for assuming that the delay of a given train (or other mode of transport, ...), at a given location and time is
independent of the delays for all other trains, locations, and times. Even if our experience tells us that this is most of the time not the case.
Also, you must assume that you have no real-time delays information at the time you plan your journey, which limits the benefits you could gain by assuming such a dependency.

##### 3 - Q: Can I take advantage of the fact that a connection departs late most of the time to allow a plan that would otherwise not be possible according to the official schedule.
* *A*: You may discover that you could take advantage of connections that have a high probability of departing late.
However, this is not recommended, or it should come with a warning.
Imagine from a user experience perspective, how would you react if you are being proposed an impossible plan in which a transfer is scheduled to depart before you arrive?
Furthermore, who would you blame if the plan fails: the planner that came up with a theoretically infeasible plan, or the operator who respected their schedule?

----

python


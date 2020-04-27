<!DOCTYPE html>
<!--

-->
<html lang="en">
    <head>
        <link rel="stylesheet" href="style.css">
        <title>Mars' Weather</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script type="text/javascript" src="automation.js"></script>
    </head>
    <body>
    <div style="background-image">

        <h1>Mars' Weather</h1>

    <figure class="rightside">
    <img class="marsRo" alt="Mars Rover." width="712" height="406" src="insight.jpg">
        <figcaption>The Insight Mars Rover</figcaption>
    </figure>

        <h2>Data Information</h2>
        <p>
            Data for the Mars Weather website was gathered from NASA’s “InSight: Mars Weather Service API.”
            The Insight rover takes continuous weather measurements of the Martian atmosphere in Elysium Planitia,
            a large plain near Mars’s equator. The API provides daily changes in weather in Elysium Planitia
            and hourly changes of winddirection and wind sample count. Below are some metrics gathered based on the
            data collected from the API.
        </p>




        <div class="grid-container1">
          <div class="grid-item1"><p>Data Last Updated: <?php include('time.txt'); ?><p></div>
          <div class="grid-item1"><p>Current Season On Mars: <?php include('season.txt'); ?></p></div>
          <div class="grid-item1"><p>Current Sol: <?php include('sol.txt'); ?></p></div>
        </div>

        <article id="wind">
            <h2> Wind Rose Plots</h2>
            <figure class="plot">
                <img class="roundedCornerSmall" width="800" height="800" alt="Wind Data" src="windrose.png">
                <figcaption>
                    This is a wind rose, which contains information about the direction, frequency,
                    and strength of wind over the last three Sols. The area in each radial histogram bin indicates
                    the frequency of wind pointing in that direction, and the counts (indicated by the color) are a proxy
                    for the strength of the wind when it was blowing that direction.
                </figcaption>
            </figure>
            <h2> Temperature/Pressure/Wind Plots</h2>
            <figure class="plot">
                <img class="roundedCornerSmall" height="800" width="1000" alt="Temperature Date" src="stackplot.png">
                <figcaption>
                    This is the average temperature, pressure, and wind speed over the past few weeks. The
                    red line is our forecast. Samples are taken once per day.
                </figcaption>
            </figure>
        </article>


        <h2>About the Team</h2>
        <div class="grid-container2">
          <div class="grid-item2"><p>Garrett Faucher, a junior Computer Science major and Astronomy minor. He likes to look at stars and bike.</p></div>
          <div class="grid-item2"><p>Eliot Heinrich, a senior physics and computer science major. He is a ballroom and swing dancer. </p></div>
          <div class="grid-item2"><p>Thomas Sheboy, a junior Computer Science major and Mathematics minor. He likes to play guitar.</p></div>
          <div class="grid-item2"><p>Stanhope Nwosu, a senior Computer Science major and Statistic minor. They like to play the trumpet and enjoys beaches.</p></div>
        </div>
        <p>
            In this project, we wanted to inform clients and others of Mars' weather.
            We decided on this project because we wanted to explore more information on Mars and
            inform others in a simplified local form. We began by extracting and organizing data on
            Mars' weather measurements and compiling them into graphs that could be easily
            understood and interpreted. The data we are making use of is from NASA's MARS InSight Mission.
            We are also utilizing this data to forecast future Mars' weather, by a day or so.
        </p>
        <p>
            Final Project for CS205 at The University of Vermont
        </p>
    </div>
    </body>
</html>

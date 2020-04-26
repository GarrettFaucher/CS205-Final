<!DOCTYPE html>
<!--

-->
<html lang="en">
    <head>
        <link rel="stylesheet" href="style.css">
        <title>Mars Weather</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script type="text/javascript" src="automation.js"></script>
    </head>
    <body>
    <div style="background-image: url('background.jpg'); background-repeat: no-repeat; background-attachment:fixed; background-size: 100% 100%; cursor: url('cursor.jpg'), auto; padding-right: 200px; padding-left: 200px;">

        <h1><strong></strong></h1>
        <h1><strong>Mars Weather</strong></h1>

    <figure class="rightside">
    <img class="marsRo" alt="Mars Rover." width="712" height="406" src="insight.jpg">
        <figcaption><strong>The Insight Mars Rover</strong></figcaption>
    </figure>

        <h2><strong>Data Information</strong></h2>
        <p><strong>
            Data for the Mars Weather website was gathered from NASA’s “InSight: Mars Weather Service API.”
            The Insight rover takes continuous weather measurements of the Martian atmosphere in Elysium Planitia,
            a large plain near Mars’s equator.
        </strong></p>

        <h2><strong>Data Last Updated: <?php include('time.txt'); ?></strong></h2>
        <p><strong>Current Season On Mars: <?php include('season.txt'); ?></strong></p>
        <p><strong>Current Sol: <?php include('sol.txt'); ?></strong></p>

        <article id="wind">
            <h2><strong> Wind Rose Plots</strong></h2>
            <figure class="plot">
                <img class="roundedCornerSmall" width="800" height="800" alt="Wind Data" src="windrose.png">
                <figcaption><strong>
                    This is a wind rose, which contains information about the direction, frequency,
                    and strength of wind over the last three Sols. The area in each radial histogram bin indicates
                    the frequency of wind pointing in that direction, and the counts (indicated by the color) are a proxy
                    for the strength of the wind when it was blowing that direction.
                </strong></figcaption>
            </figure>
            <h2> <strong>Temperature/Pressure/Wind Plots</strong></h2>
            <figure class="plot">
                <img class="roundedCornerSmall" height="800" width="1000" alt="Temperature Date" src="stackplot.png">
                <figcaption><strong>
                    This is the average temperature, pressure, and wind speed over the past few weeks. The
                    red line is our forecast. Samples are taken once per day.
                </strong></figcaption>
            </figure>
        </article>

        <h3><strong>About the Team</strong></h3>
        <p><strong>Garrett Faucher, a junior Computer Science major and Astronomy minor. He likes to look at stars and bike.</strong></p>
        <p><strong>Eliot Heinrich, a senior physics and computer science major. He is a ballroom and swing dancer. </strong></p>
        <p><strong>Thomas Sheboy, a junior Computer Science major and Mathematics minor. He likes to play guitar.</strong></p>
        <p><strong>Stanhope Nwosu, a senior Computer Science major and Statistic minor. They like to play the trumpet and enjoys beaches.</strong></p>
        <p><strong></strong></p>
        <p1><strong>
            In this project, we wanted to inform clients and others of Mars' weather.
            We decided on this project because we wanted to explore more information on Mars and
            inform others in a simplified local form. We began by extracting and organizing data on
            Mars' weather measurements and compiling them into graphs that could be easily
            understood and interpreted. The data we are making use of is from NASA's MARS InSight Mission.
            We are also utilizing this data to forecast future Mars' weather, by a day or so.
        </strong></p1>
    </div>
    </body>
</html>

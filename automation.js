// Runs genplots.py on each 30th minute of the hour

cron.schedule(“30 * * * *”, function () {
  console.log(“Running Cron Job”);
  const exec = require('child_process').exec, child;
  const cliCommand = exec('python3 genplots.py');
  cliCommand.stdout.on('data', (data)=>{
      console.log(data);
  });
});

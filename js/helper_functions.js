/*----------------------------------------------------------------------*/
/* The name says it all                                                 */
/*----------------------------------------------------------------------*/
/*-----------------------------------------------------------------------*/
/* Subroutines                                                           */
/*-----------------------------------------------------------------------*/
function filterTasks(occ_code)
{
    var results = [];
    for (var i = 0; i < taskData.length; i++)
    {
	if (taskData[i].fedscope_occ_code == occ_code)
	{
	    newObject = {};
	    newObject = taskData[i];
	    newObject.hours = freq2hours(newObject.category);
	    results.push(newObject);
	}
    }
    return results;
}

/*-----------------------------------------------------------------------*/
/* Convert frequency categories to hours                                 */
/*-----------------------------------------------------------------------*/
function freq2hours(category)
{
    var hours = 0;
    switch(category)
    {
	case "1":
	hours = .5;
	break;
	case "2":
	hours = 1;
	break;
	case "3":
	hours = 12;
	break;
	case "4":
	hours = 52;
	break;
	case "5":
	hours = 260;
	break;
	case "6":
	hours = 520;
	break;
	case "7":
	hours = 1043;
	break;
	default:
	console.log("Unknown frequency category: " + category);
	break;
    }
    return hours;
}

/*-----------------------------------------------------------------------*/
/* refresh the pie chart when you pass in a new occ code                 */
/*-----------------------------------------------------------------------*/
function refreshPie(newOccCode)
{
    filteredTaskData = [];
    filteredTaskData = filterTasks(newOccCode);
    pieChartHandle.data(filteredTaskData)
	.draw();
    $('#pie_heading').html("Hours per task for occupation " + newOccCode);
}


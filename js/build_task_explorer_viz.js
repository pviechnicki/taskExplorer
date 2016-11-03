/*-----------------------------------------------------------------------*/
/* Build pie chart and tree map of tasks and occupations, link them      */
/*-----------------------------------------------------------------------*/

/*-----------------------------------------------------------------------*/
/* Globals                                                               */
/*-----------------------------------------------------------------------*/
var tableHandle = {};
var pieChartHandle = {};
var fakeData = [
    {"task_id": "task 1", "hours": 2},
    {"task_id": "task 2", "hours": 1},
    {"task_id": "task 3", "hours": 5}
];
//Empty array of objects to hold filtered data
var filteredTaskData = [];


/*-----------------------------------------------------------------------*/
/* Main functionality                                                    */
/*-----------------------------------------------------------------------*/
//Bind the occupations data to the table
tableHandle = $('#task_table').dynatable(
    {
	table:
	{
	    defaultColumnIdStyle: 'underscore'
	},
	dataset:
	{
	    records: occupations,
	    perPageDefault: 3
	}

    }
);
/*-----------------------------------------------------------------------*/
/* Bind function to click events for table of occupations                */
/* refresh the pie chart when you click a row                            */
/*-----------------------------------------------------------------------*/
tableHandle.dynatable().on('click', 'tr', function(e)
			   {
			       refreshPie(e.currentTarget.firstChild.innerHTML);
			   });

var dummyOcc = "0006";
filteredTaskData = filterTasks(dummyOcc);

pieChartHandle = d3plus.viz()
    .container("#task_pie_div")
    .type("pie")
    .data(filteredTaskData)
    .id("task_id")
    .size("hours")
    .tooltip("task_desc")
    .text({"value": function(d) {return (d.task_desc.split(" ")[0] + "..."); }})
    .draw();

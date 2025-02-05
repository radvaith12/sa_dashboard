import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import smtplib
import warnings
from decimal import Decimal
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

def connect_to_database():
    dbname = "automation"
    user = "traders_read"
    password = "scalp123$"
    host = "10.5.1.20"
    port = "5432"

    try:
        connection = psycopg2.connect(
            dbname=dbname, user=user, password=password, host=host, port=port
        )
        return connection
    except Exception as e:
        print(f"Error: Unable to connect to the database - {e}")
        return None


def run_query1(logtime1, logtime2):
    global results
    global columns
    sql_query = f"""SELECT spreadarb_filteredlean.opportunityid,ratiocount, spreadarb_orderlean.logtime, spreadarb_orderlean.underlying, spreadarb_orderlean.strategyname, event, min(theovalue) as theovalue, theocalculated, spreadarb_orderlean.exchange, spreadarb_orderlean.price, lastfillqty, spreadarb_orderlean.strategyid, instrument, spreadarb_orderlean.side, lastfillpip, max(qty) as qty, max(delta) as delta, max(mpedge) as mpedge, max(edgepercentage) as edgepercentage, xdays, spreadtype, lastxday, max(impspread) as impspread, max(valr) as valr, max(spreadvalue) as spreadvalue
FROM spreadarb_orderlean
INNER JOIN spreadarb_filteredlean ON spreadarb_orderlean.opportunityid = spreadarb_filteredlean.opportunityid
AND event IN ('FILLED','PARTIAL')
AND spreadarb_filteredlean.symid = spreadarb_orderlean.instrument
AND spreadarb_filteredlean.instanceid = spreadarb_orderlean.instanceid
WHERE spreadarb_orderlean.logtime >= '{logtime1}' AND spreadarb_orderlean.logtime < '{logtime2}'
AND poseffect = 'OPEN'
AND spreadarb_filteredlean.updatetype IN ('ADD','UPDATE')
GROUP BY spreadarb_filteredlean.opportunityid,ratiocount, spreadarb_orderlean.logtime, theocalculated, spreadarb_orderlean.underlying, event, spreadarb_orderlean.exchange, spreadarb_orderlean.strategyname, spreadarb_orderlean.price, lastfillqty, spreadarb_orderlean.strategyid, instrument, spreadarb_orderlean.side, lastfillpip, xdays, spreadtype, lastxday
"""
    db_connection = connect_to_database()

    if db_connection:
        cursor = db_connection.cursor()
        cursor.execute(sql_query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()


def run_query2(logtime1, logtime2):
    global results
    global columns
    sql_query = f"""SELECT spreadarb_filteredlean.opportunityid,ratiocount, instrument, spreadarb_orderlean.logtime, spreadarb_orderlean.underlying, event, spreadarb_orderlean.strategyname, min(theovalue) as theovalue, theocalculated, spreadarb_orderlean.exchange, spreadarb_orderlean.price, lastfillqty, spreadarb_orderlean.strategyid, instrument, spreadarb_orderlean.side, lastfillpip, max(qty) as qty, max(delta) as delta, max(mpedge) as mpedge, max(edgepercentage) as edgepercentage, xdays, spreadtype, lastxday, max(impspread) as impspread, max(valr) as valr, max(spreadvalue) as spreadvalue
FROM spreadarb_orderlean
INNER JOIN spreadarb_filteredlean ON spreadarb_orderlean.opportunityid = spreadarb_filteredlean.opportunityid
AND event IN ('FILLED','PARTIAL')
AND spreadarb_filteredlean.symid = spreadarb_orderlean.instrument
AND spreadarb_filteredlean.instanceid = spreadarb_orderlean.instanceid
WHERE spreadarb_orderlean.logtime >= '{logtime1}' AND spreadarb_orderlean.logtime < '{logtime2}'
AND poseffect = 'CLOSE'
AND spreadarb_filteredlean.updatetype IN ('ADD','UPDATE')
GROUP BY spreadarb_filteredlean.opportunityid, spreadarb_orderlean.logtime,ratiocount,instrument, event, spreadarb_orderlean.underlying, spreadarb_orderlean.strategyname, theocalculated, spreadarb_orderlean.exchange, spreadarb_orderlean.price, lastfillqty, spreadarb_orderlean.strategyid, instrument, spreadarb_orderlean.side, lastfillpip, xdays, spreadtype, lastxday
"""

    db_connection = connect_to_database()

    if db_connection:
        cursor = db_connection.cursor()
        cursor.execute(sql_query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()


def run_query3(logtime1, logtime2):
    global results
    global columns
    sql_query = f"""select sum(lastfillqty),strategyname from spreadarb_orderlean
WHERE logtime >= '{logtime1}' AND logtime < '{logtime2}'
and event in ('FILLED','PARTIAL')
and poseffect = 'OPEN'
group by strategyname
order by strategyname desc;;"""

    db_connection = connect_to_database()

    if db_connection:
        cursor = db_connection.cursor()
        cursor.execute(sql_query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()


def run_query4(logtime1, logtime2):
    global results
    global columns
    sql_query = f"""select count(event),strategyname from spreadarb_orderlean
WHERE logtime >= '{logtime1}' AND logtime < '{logtime2}'
and event in ('OPEN','AMENDED')
and poseffect = 'OPEN'
group by strategyname
order by strategyname desc;;"""

    db_connection = connect_to_database()

    if db_connection:
        cursor = db_connection.cursor()
        cursor.execute(sql_query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()


def run_query4000(logtime1, logtime2):
    global results
    global columns
    sql_query = f"""select count(distinct(instrument)) from spreadarb_orderlean
where logtime >=  '{logtime1}'
AND logtime < '{logtime2}'
and event in ('OPEN')
and exchange = 'ARCA'
order by count desc;;"""

    db_connection = connect_to_database()

    if db_connection:
        cursor = db_connection.cursor()
        cursor.execute(sql_query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()

        
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 🎯 **Ask user for logtime1 and logtime2**
# 🎯 **Ask user for logtime1 and logtime2**
st.sidebar.header("📅 Select Date & Time Range")

# Set Default Date (Yesterday to Today)
default_start_date = datetime.today().date() - timedelta(days=7)  # Yesterday
default_end_date = datetime.today().date()  # Today

# Set Default Time (US Stock Market Hours)
default_start_time = datetime.strptime("09:30:00", "%H:%M:%S").time()  # 9:30 AM
default_end_time = datetime.strptime("16:00:00", "%H:%M:%S").time()  # 4:00 PM

# Date Input (With Defaults)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", default_end_date)

# Time Input (With Defaults)
start_time = st.sidebar.time_input("Start Time", default_start_time)
end_time = st.sidebar.time_input("End Time", default_end_time)

# Convert to correct format for SQL queries
logtime1 = f"{start_date} {start_time}.000"
logtime2 = f"{end_date} {end_time}.000"

st.sidebar.write(f"🔍 Selected Log Time Range:")
st.sidebar.write(f"**Start:** {logtime1}")
st.sidebar.write(f"**End:** {logtime2}")

#today_date = datetime.today().date()
#tomorrow_date = today_date + timedelta(days=1)
#logtime1 = str(today_date) + ' 00:00:00.000'
#logtime2 = str(tomorrow_date) + ' 00:00:00.000'
#logtime1 = '2025-01-27 00:00:00.000'
#logtime2 = '2025-02-01 00:00:00.000'

fill_open = pd.DataFrame()
fill_close = pd.DataFrame()
fill_open_temp = pd.DataFrame()
fill_close_temp = pd.DataFrame()
global columns, results

run_query1(logtime1, logtime2)
fill_open_temp = pd.DataFrame(results, columns=columns)

run_query2(logtime1, logtime2)
fill_close_temp = pd.DataFrame(results, columns=columns)

#run_query3(logtime1, logtime2)
#fill_rate = pd.DataFrame(results, columns=columns)

#run_query4(logtime1, logtime2)
#open_rate = pd.DataFrame(results, columns=columns)

fill_open = pd.concat([fill_open, fill_open_temp])
fill_close = pd.concat([fill_close, fill_close_temp])

fill_open['logtime'] = fill_open['logtime'].astype(str)
fill_open['date'] = fill_open['logtime'].str.split(' ', expand=True)[0]
fill_open['time'] = fill_open['logtime'].str.split(' ', expand=True)[1]

fill_close['logtime'] = fill_close['logtime'].astype(str)
fill_close['date'] = fill_close['logtime'].str.split(' ', expand=True)[0]
fill_close['time'] = fill_close['logtime'].str.split(' ', expand=True)[1]

fill_open = fill_open.sort_values(by='logtime')
fill_close = fill_close.sort_values(by='logtime')

fill_open = fill_open.reset_index(drop=True)
fill_close = fill_close.reset_index(drop=True)
fill_open_temp11 = fill_open.copy()
fill_close_temp11 = fill_close.copy()

fill_open = fill_open.reset_index(drop=True)
fill_close = fill_close.reset_index(drop=True)

#fill_open = fill_open[fill_open['underlying'] == 'MSTR'].reset_index(drop=True)
#fill_close = fill_close[fill_close['underlying'] == 'MSTR'].reset_index(drop=True)

filldata = pd.DataFrame()
count = 0
for i in fill_open['opportunityid'].unique():
    fillopentemp = fill_open[fill_open['opportunityid'] == i]
    fillopentemp = fillopentemp.reset_index(drop=True)
    for j in fillopentemp['exchange'].unique():
        fillopentemp1 = fillopentemp[fillopentemp['exchange'] == j]
        fillopentemp1 = fillopentemp1.reset_index(drop=True)
        filldata = pd.concat([filldata, fillopentemp1.iloc[[0]]])
        filldata = filldata.reset_index(drop=True)
        filldata.loc[count, 'lastfillpip'] = np.abs(np.sum(fillopentemp1['lastfillpip']))
        filldata.loc[count, 'lastfillqty'] = np.sum(fillopentemp1['lastfillqty'])
        filldata.loc[count, 'price'] = np.abs(np.sum(fillopentemp1['price'] * fillopentemp1['lastfillqty']))
        filldata.loc[count, 'price'] = filldata.loc[count, 'price'] / filldata.loc[count, 'lastfillqty']
        count += 1

filldatamain = pd.DataFrame()
for i in filldata['opportunityid'].unique():
    temp = filldata[filldata['opportunityid'] == i]
    temp = temp.reset_index(drop=True)
    openfillqty = np.sum(temp['lastfillqty'])
    temp1 = fill_close[fill_close['opportunityid'] == i]
    closefillqty = np.sum(temp1['lastfillqty'])
    for j in range(len(temp)):
        temp.loc[j, 'valr'] = temp.loc[j, 'lastfillqty'] * (closefillqty / openfillqty)
        temp.loc[j, 'mpedge'] = np.abs(np.sum(temp1['price'] * temp1['lastfillqty']) / np.sum(temp1['lastfillqty']))
    filldatamain = pd.concat([filldatamain, temp])

filldatamain = filldatamain.reset_index(drop=True)
filldatamain = filldatamain.dropna().reset_index(drop=True)

filldatamain['edge'] = 0
filldatamain['sizefilled'] = 0
for i in range(len(filldatamain)):
    filldatamain.loc[i, 'edge'] = np.abs(filldatamain.loc[i, 'price'] - filldatamain.loc[i, 'mpedge'])
    filldatamain.loc[i, 'sizefilled'] = np.abs(filldatamain.loc[i, 'valr'])

filldatamain['theoedge'] = 0
filldatamain['pnl'] = 0
for i in range(len(filldatamain)):
    if filldatamain.loc[i, 'theovalue'] < 0:
        filldatamain.loc[i, 'theovalue'] *= -1
    if filldatamain.loc[i, 'side'] == 'BUY':
        filldatamain.loc[i, 'theoedge'] = filldatamain.loc[i, 'theovalue'] - filldatamain.loc[i, 'price']
    else:
        filldatamain.loc[i, 'theoedge'] = filldatamain.loc[i, 'price'] - filldatamain.loc[i, 'theovalue']


    filldatamain.loc[i, 'pnl'] = (filldatamain.loc[i, 'edge'] + Decimal(abs(filldatamain.loc[i, 'lastfillpip']))) * \
                             Decimal(int(filldatamain.loc[i, 'sizefilled']))

filldatamain['delta'] = filldatamain['delta'].astype(float)
filldatamain['lastfillpip'] = filldatamain['lastfillpip'].astype(float)
filldatamain['deltaround'] = np.round(filldatamain['delta'] / 5) * 5
filldatamain['xdaysround'] = np.round(filldatamain['xdays'] / 5) * 5

filldatamain['executions'] = filldatamain['lastfillqty']
filldatamain['contracts'] = filldatamain['lastfillqty']*filldatamain['ratiocount']
filldatamain['marketshare'] = filldatamain['executions']/filldatamain['qty']
filldatamain['marketshare'] = filldatamain['marketshare'].clip(upper=1)

filldatamain['theoedge_impspread'] = filldatamain['theoedge']/filldatamain['impspread']

# Rename a single column
filldatamain.rename(columns={'logtime': 'timestamp'}, inplace=True)


# Step 1: Ensure 'timestamp' is in datetime format
filldatamain['timestamp'] = pd.to_datetime(filldatamain['timestamp'], errors='coerce')

# Step 2: Extract 'date' and round 'timestamp' to the nearest 30-minute interval
filldatamain['date'] = filldatamain['timestamp'].dt.date
filldatamain['half_hour'] = filldatamain['timestamp'].dt.floor('30min')

# Step 3: Group data by date and 30-minute windows
grouped_daily = filldatamain.groupby(['date', 'half_hour']).agg(total_made=('pnl', 'sum')).reset_index()

# Step 4: Calculate %PnL per day
grouped_daily['%pnl'] = grouped_daily.groupby('date')['total_made'].transform(lambda x: x / x.sum())

# Step 5: Convert %PnL to numeric
grouped_daily['%pnl'] = pd.to_numeric(grouped_daily['%pnl'], errors='coerce').fillna(0)

# Step 6: Calculate mean and variance of %PnL across days for each 30-minute window
grouped = grouped_daily.groupby('half_hour').agg(
    mean_pnl=('total_made', 'mean'),
    variance_pnl=('%pnl', 'var')  # Variance of normalized PnL
).reset_index()

# Step 7: Add 'time' column for visualization
grouped['time'] = grouped['half_hour'].dt.time


# Ensure total_made is numeric
grouped_daily['total_made'] = pd.to_numeric(grouped_daily['total_made'], errors='coerce')

# Handle any NaN values in total_made
grouped_daily['total_made'].fillna(0, inplace=True)

# Ensure %PnL is already numeric (as per debug output)
grouped_daily['%pnl'] = grouped_daily['%pnl'].astype(float)

# Step 3: Calculate mean and variance of %PnL across days for each 30-minute window
grouped = grouped_daily.groupby('half_hour').agg(
    mean_pnl=('total_made', 'mean'),  # Now numeric
    variance_pnl=('%pnl', 'var')  # Already numeric
).reset_index()

# Step 4: Add 'time' column for visualization
grouped['time'] = grouped['half_hour'].dt.time

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Ensure 'timestamp' is in datetime format
filldatamain['timestamp'] = pd.to_datetime(filldatamain['timestamp'], errors='coerce')

# Extract 'date' and round 'timestamp' to the nearest 30-minute interval
filldatamain['date'] = filldatamain['timestamp'].dt.date
filldatamain['half_hour'] = filldatamain['timestamp'].dt.floor('30min')



fillgoals = filldatamain[['date','pnl','contracts','executions','marketshare','lastfillpip']]
fillgoalsmain = fillgoals[:len(fillgoals['date'].unique())]
count = 0
for i in fillgoals['date'].unique():

    fillgoals_temp = fillgoals[fillgoals['date'] == i].reset_index(drop = True)
    fillgoalsmain['date'][count] = fillgoals_temp['date'][0]
    fillgoalsmain['pnl'][count] = np.sum(fillgoals_temp['pnl'])
    fillgoalsmain['contracts'][count] = np.sum(fillgoals_temp['contracts'])
    fillgoalsmain['executions'][count] = np.sum(fillgoals_temp['executions'])
    fillgoalsmain['marketshare'][count] = np.mean(fillgoals_temp['marketshare'])
    fillgoalsmain['lastfillpip'][count] = np.sum(fillgoals_temp['lastfillpip'])
    count += 1



import streamlit as st

# 🎯 **Tab for Data Table Display**
#st.subheader("PnL, Contracts, Executions, and Last Fill Pip")

# 🎯 **Show the DataFrame inside Streamlit**
#st.dataframe(fillgoalsmain[['date','pnl','contracts','executions','lastfillpip']])


fill_plot = fillgoalsmain[['date','pnl','contracts','executions','lastfillpip']].copy()
pnl_sum = np.sum(fillgoalsmain['pnl'])
contracts_sum = np.sum(fillgoalsmain['contracts'])
executions_sum = np.sum(fillgoalsmain['executions'])
fill_plot['pnl'] /= pnl_sum/100
fill_plot['contracts'] /= contracts_sum/100
fill_plot['executions'] /= executions_sum/100
fill_plot['pnl'] = fill_plot['pnl'].astype(float)
fillgoalsmain['pnl'] = 100*fillgoalsmain['pnl'].astype(float)
fillgoalsmain['contracts'] = fillgoalsmain['contracts'].astype(float)
fillgoalsmain['executions'] = fillgoalsmain['executions'].astype(float)

overview = pd.DataFrame(columns = ['Parameter','Average','Variance'])
overviewtemp = pd.DataFrame([['PnL',fillgoalsmain['pnl'].mean(),fillgoalsmain['pnl'].std()]],columns = ['Parameter','Average','Variance'])
overview = pd.concat([overview,overviewtemp])
overviewtemp = pd.DataFrame([['Contracts',fillgoalsmain['contracts'].mean(),fillgoalsmain['contracts'].std()]],columns = ['Parameter','Average','Variance'])
overview = pd.concat([overview,overviewtemp])
overviewtemp = pd.DataFrame([['Executions',fillgoalsmain['executions'].mean(),fillgoalsmain['executions'].std()]],columns = ['Parameter','Average','Variance'])
overview = pd.concat([overview,overviewtemp])

overview.reset_index(drop = True)
overview[['Average', 'Variance']] = overview[['Average', 'Variance']].round(2)
# Group data by date and 30-minute windows, including sum of `sizefilled`
grouped_daily = filldatamain.groupby(['date', 'half_hour']).agg(
    total_made=('pnl', 'sum'),
    sizefilled_sum=('sizefilled', 'sum')
).reset_index()

# Convert to numeric
grouped_daily['total_made'] = pd.to_numeric(grouped_daily['total_made'], errors='coerce')
grouped_daily['sizefilled_sum'] = pd.to_numeric(grouped_daily['sizefilled_sum'], errors='coerce')

# Fill NaN values
grouped_daily.fillna(0, inplace=True)

# Aggregate data by half_hour to calculate mean and variance
grouped_daily['half_hour'] = grouped_daily['half_hour'].dt.time
grouped = grouped_daily.groupby('half_hour').agg(
    mean_pnl=('total_made', 'mean'),
    variance_pnl=('total_made', 'var'),
    mean_sizefilled=('sizefilled_sum', 'mean'),
    variance_sizefilled=('sizefilled_sum', 'var'),
    total_sizefilled=('sizefilled_sum', 'sum')
).reset_index()

# Add 'time' column for visualization
grouped['time'] = grouped['half_hour'].astype(str)

# Normalize total_sizefilled for gradient coloring
norm_sizefilled = (grouped['total_sizefilled'] - grouped['total_sizefilled'].min()) / (
    grouped['total_sizefilled'].max() - grouped['total_sizefilled'].min()
)


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 🎯 **Create Seven Tabs for Visualization**
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview Table",
    "📈 % Total PnL",
    "📈 PnL Over 30-Minute Windows",
    "📈 PnL Over Time",
    "📊 Contracts & Executions",
    "📊 Market Share vs PnL",
    "📊 Last Fill Pip vs PnL"
])

# 📊 **Tab 1: Overview Table**
with tab1:
    st.subheader("Overview Table")
    st.dataframe(fillgoalsmain[['date', 'pnl', 'contracts', 'executions', 'lastfillpip']])

# 📈 **Tab 2: Multi-Axis Plot - % Total PnL**    
with tab2:
    st.subheader("PnL, Contracts, and Executions Over Time (Multi-Axis)")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Ensure 'date' is a string for plotting
    fill_plot['date'] = fill_plot['date'].astype(str)

    # Plot PnL on the first y-axis
    ax1.plot(fill_plot['date'], fill_plot['pnl'], label='PnL', color='blue', marker='o')
    ax1.set_ylabel('PnL', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')

    # Set x-axis ticks properly
    ax1.set_xticks(range(len(fill_plot['date'])))
    ax1.set_xticklabels(fill_plot['date'], rotation=90, fontsize=10, ha="center")  # Rotate to vertical

    # Create a second y-axis for Contracts
    ax2 = ax1.twinx()
    ax2.plot(fill_plot['date'], fill_plot['contracts'], label='Contracts', color='green', marker='o')
    ax2.set_ylabel('Contracts', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')

    # Create a third y-axis for Executions (offset to the right)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 80))  # Increased spacing
    ax3.plot(fill_plot['date'], fill_plot['executions'], label='Executions', color='orange', marker='o')
    ax3.set_ylabel('Executions', color='orange', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='orange')

    # Adjust layout to avoid overlap
    fig.subplots_adjust(right=0.75)

    # Add legend manually
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')

    # Add title and format plot
    ax1.set_xlabel("Date", fontsize=12)
    plt.title("PnL, Contracts, and Executions Over Time (%)", fontsize=14)
    plt.xticks(rotation=90, fontsize=10, ha="center")  # Ensure labels are vertical

    fig.tight_layout()
    st.pyplot(fig)

# 📈 **Tab 3: PnL Over 30-Minute Windows**
with tab3:
    colors = sns.color_palette("Blues", len(grouped))
    gradient_colors = [colors[int(val * (len(colors) - 1))] for val in norm_sizefilled]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar plot with gradient colors for mean_pnl
    bars = ax1.bar(grouped['time'], grouped['mean_pnl'], color=gradient_colors, alpha=0.8, label='Mean PnL')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Mean PnL', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(range(len(grouped['time'])))
    ax1.set_xticklabels(grouped['time'], rotation=90, fontsize=10)

    # Line plot for variance of PnL on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(grouped['time'], grouped['variance_pnl'], color='red', marker='o', label='Variance of PnL')
    ax2.set_ylabel('Variance of PnL', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add color bar for the gradient
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=grouped['total_sizefilled'].min(), vmax=grouped['total_sizefilled'].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, pad=0.15, location='right')
    cbar.set_label('Total Size Filled', fontsize=12)

    # Adjust layout
    plt.title('Mean and Variance of PnL over 30-Minute Windows (Gradient by Total Size Filled)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    st.pyplot(fig)

# 📈 **Tab 4: PnL Over Time**
with tab4:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fillgoalsmain['date'], fillgoalsmain['pnl'], marker='o', color='blue')
    ax.set_title('PnL Over Time', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('PnL', fontsize=12)
    ax.set_xticklabels(fillgoalsmain['date'], rotation=45, fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

# 📊 **Tab 5: Contracts & Executions Over Time**
with tab5:
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.4  # Width of bars
    x = range(len(fillgoalsmain))  # X-axis positions
    ax.bar(x, fillgoalsmain['contracts'], width=width, label='Contracts', color='skyblue', edgecolor='black')
    ax.bar([p + width for p in x], fillgoalsmain['executions'], width=width, label='Executions', color='orange', edgecolor='black')
    ax.set_title('Contracts and Executions Over Time', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(fillgoalsmain['date'], rotation=45, fontsize=10)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# 📊 **Tab 6: Market Share vs PnL**
with tab6:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(fillgoalsmain['marketshare'], fillgoalsmain['pnl'], c='green', alpha=0.7, edgecolor='black')
    ax.set_title('Market Share vs PnL', fontsize=14)
    ax.set_xlabel('Market Share', fontsize=12)
    ax.set_ylabel('PnL', fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

# 📊 **Tab 7: Last Fill Pip vs PnL**
with tab7:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(fillgoalsmain['lastfillpip'], fillgoalsmain['pnl'], c='red', alpha=0.7, edgecolor='black')
    ax.set_title('Last Fill Pip vs PnL', fontsize=14)
    ax.set_xlabel('Last Fill Pip', fontsize=12)
    ax.set_ylabel('PnL', fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)





# Calculate total PnL
total_pnl = filldatamain['pnl'].sum()

# Group by 'underlying' and calculate %PnL
pnl_underlying = (
    filldatamain.groupby('underlying')['pnl']
    .sum()
    .reset_index()
    .rename(columns={'pnl': 'total_pnl'})
)

# Calculate %PnL
pnl_underlying['%pnl'] = (pnl_underlying['total_pnl'] / total_pnl) * 100

# Sort by %PnL in descending order
pnl_underlying = pnl_underlying.sort_values(by='%pnl', ascending=False).reset_index(drop=True)

import streamlit as st
import matplotlib.pyplot as plt

# 📊 **Create Tabs for PnL Contribution**
tab1, tab2 = st.tabs([
    "📊 PnL Contribution - Pie Chart",
    "📈 PnL Contribution - Bar Chart"
])

# **Calculate total PnL**
total_pnl = filldatamain['pnl'].sum()

# **Group by 'underlying' and calculate %PnL**
pnl_underlying = (
    filldatamain.groupby('underlying')['pnl']
    .sum()
    .reset_index()
    .rename(columns={'pnl': 'total_pnl'})
)

# **Calculate %PnL**
pnl_underlying['%pnl'] = (pnl_underlying['total_pnl'] / total_pnl) * 100

# **Sort by %PnL in descending order**
pnl_underlying = pnl_underlying.sort_values(by='%pnl', ascending=False).reset_index(drop=True)

# **Tab 1: PnL Contribution - Pie Chart**
with tab1:
    st.subheader("PnL Contribution by Underlying (Grouped) - Pie Chart")

    # Define threshold for grouping smaller categories
    threshold = 1.5  # Percentage below which categories will be grouped as "Others"
    
    # Group small categories under 'Others'
    pnl_underlying['category'] = pnl_underlying['%pnl'].apply(lambda x: 'Others' if x < threshold else None)
    pnl_underlying.loc[pnl_underlying['category'].isna(), 'category'] = pnl_underlying['underlying']

    # Aggregate the 'Others' category
    grouped_data = pnl_underlying.groupby('category', as_index=False)['%pnl'].sum()

    # Create Pie Chart
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(
        grouped_data['%pnl'],
        labels=grouped_data['category'],
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.tab20.colors,
        wedgeprops={'edgecolor': 'black'}
    )
    
    # Set Title
    ax.set_title('PnL Contribution by Underlying (Grouped)', fontsize=14)

    # Display Pie Chart in Streamlit
    st.pyplot(fig)


# **Tab 2: PnL Contribution - Bar Chart**
with tab2:
    st.subheader("PnL Contribution by Underlying (Grouped) - Bar Chart")

    # Define threshold for grouping smaller categories
    threshold = 0.5  # Percentage below which categories will be grouped as "Others"
    
    # Group small categories under 'Others'
    pnl_underlying['category'] = pnl_underlying['%pnl'].apply(lambda x: 'Others' if x < threshold else None)
    pnl_underlying.loc[pnl_underlying['category'].isna(), 'category'] = pnl_underlying['underlying']

    # Aggregate the 'Others' category
    grouped_data = pnl_underlying.groupby('category', as_index=False)['%pnl'].sum()

    # Create Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        grouped_data['category'],
        grouped_data['%pnl'],
        color=plt.cm.tab20.colors[:len(grouped_data)],  # Use a subset of the color palette
        edgecolor='black'
    )

    # Labels & Titles
    ax.set_xlabel('Underlying', fontsize=12)
    ax.set_ylabel('%PnL Contribution', fontsize=12)
    ax.set_title('PnL Contribution by Underlying (Grouped)', fontsize=14)
    ax.set_xticklabels(grouped_data['category'], rotation=45, fontsize=10)

    # Optimize layout
    plt.tight_layout()

    # Display Bar Chart in Streamlit
    st.pyplot(fig)

spreadtype = pd.DataFrame(columns=['SpreadType', 'Mean', 'Variance'])
count = 0

for i in filldatamain['spreadtype'].unique():
    spread = filldatamain[filldatamain['spreadtype'] == i].reset_index(drop=True)
    pnl_store = np.zeros(len(spread['date'].unique()))
    count = 0

    for j in spread['date'].unique():
        spread_date = spread[spread['date'] == j]
        pnl_store[count] = np.sum(100*spread_date['pnl'])
        count += 1
    spreadtypetemp = pd.DataFrame([[i, np.mean(pnl_store), np.std(pnl_store)]], columns=['SpreadType', 'Mean', 'Variance'])
    spreadtype = pd.concat([spreadtype, spreadtypetemp], ignore_index=True)

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 🎯 **Create Tabs**
tab1, tab2 = st.tabs([
    "📊 Mean & Variance by Spread Type",
    "📉 Variance vs. Mean PnL"
])

# 📊 **Tab 1: Mean & Variance by Spread Type (Combined)**
with tab1:
    st.subheader("Mean and Variance by Spread Type")

    # Create the combined plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar plot for 'Mean'
    sns.barplot(data=spreadtype, x='SpreadType', y='Mean', palette='viridis', edgecolor='black', ax=ax1)
    ax1.set_ylabel('Mean PnL', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(spreadtype['SpreadType'], rotation=45, ha='right', fontsize=10)

    # Secondary axis for 'Variance'
    ax2 = ax1.twinx()
    ax2.plot(spreadtype['SpreadType'], spreadtype['Variance'], color='red', marker='o', label='Variance')
    ax2.set_ylabel('Variance', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')

    # Title and Layout
    plt.title("Mean and Variance by Spread Type", fontsize=14)
    fig.tight_layout()

    # 🎯 **Show the plot inside Streamlit**
    st.pyplot(fig)

# 📉 **Tab 2: Scatter Plot - Variance vs. Mean PnL**
with tab2:
    st.subheader("Variance vs. Mean PnL by Spread Type")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(spreadtype['Mean'], spreadtype['Variance'], color='blue', alpha=0.7, edgecolor='black')
    ax.set_title('Variance vs. Mean PnL by Spread Type', fontsize=14)
    ax.set_xlabel('Mean PnL', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)

    # Annotate each point with SpreadType name
    for i, txt in enumerate(spreadtype['SpreadType']):
        ax.annotate(txt, (spreadtype['Mean'][i], spreadtype['Variance'][i]), fontsize=9, alpha=0.8)

    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 🎯 **Show the scatter plot inside Streamlit**
    st.pyplot(fig)

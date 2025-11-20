import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# Show all columns
pd.set_option('display.max_columns', None)
# Show all rows
pd.set_option('display.max_rows', None)

# Load data
df = pd.read_csv("data/speech_paragraphs_with_predictions.csv")

# Create a dictionary mapping claim codes to their text descriptions
subsubclaim_map = {
    "claim_0_0_0": "No Claim detected",
    "claim_1_0_0": "Global warming is not happening",
    "claim_1_1_0": "Ice/permafrost/snow cover isn't melting",
    "claim_1_1_1": "Antarctica is gaining ice/not warming",
    "claim_1_1_2": "Greenland is gaining ice/not melting",
    "claim_1_1_3": "Arctic sea ice isn't vanishing",
    "claim_1_1_4": "Glaciers aren't vanishing",
    "claim_1_2_0": "We're heading into an ice age/global cooling",
    "claim_1_3_0": "Weather is cold/snowing",
    "claim_1_4_0": "Climate hasn't warmed/changed over the last (few) decade(s)",
    "claim_1_5_0": "Oceans are cooling/not warming",
    "claim_1_6_0": "Sea level rise is exaggerated/not accelerating",
    "claim_1_7_0": "Extreme weather isn't increasing/has happened before/isn't linked to climate change",
    "claim_1_8_0": "They changed the name from 'global warming' to 'climate change'",
    "claim_2_0_0": "Human greenhouse gases are not causing climate change",
    "claim_2_1_0": "It's natural cycles/variation",
    "claim_2_1_1": "It's the sun/cosmic rays/astronomical",
    "claim_2_1_3": "It's the ocean/internal variability",
    "claim_2_1_4": "Climate has changed naturally/been warm in the past",
    "claim_2_3_0": "There's no evidence for greenhouse effect/carbon dioxide driving climate change",
    "claim_2_3_1": "Carbon dioxide is just a trace gas",
    "claim_2_3_3": "Carbon dioxide lags/not correlated with climate change",
    "claim_2_3_4": "Water vapor is the most powerful greenhouse gas",
    "claim_2_3_6": "CO2 is not rising.",
    "claim_3_0_0": "Climate impacts/global warming is beneficial/not bad",
    "claim_3_1_0": "Climate sensitivity is low/negative feedbacks reduce warming",
    "claim_3_2_0": "Species/plants/reefs aren't showing climate impacts yet/are benefiting from climate change",
    "claim_3_2_2": "Polar bears are not in danger from climate change",
    "claim_3_2_3": "Ocean acidification/coral impacts aren't serious",
    "claim_3_3_0": "CO2 is beneficial/not a pollutant",
    "claim_3_3_1": "CO2 is plant food",
    "claim_3_4_0": "It's only a few degrees (or less)",
    "claim_3_5_0": "Climate change does not contribute to human conflict/threaten national security",
    "claim_3_6_0": "Climate change doesn't negatively impact health",
    "claim_4_0_0": "Climate solutions won't work",
    "claim_4_1_0": "Climate solutions are harmful",
    "claim_4_1_1": "Solutions increases costs",
    "claim_4_1_2": "Policy weakens security",
    "claim_4_1_3": "Solutions harm environment",
    "claim_4_1_4": "Policy creates uncertainty",
    "claim_4_1_5": "Policy limits freedom",
    "claim_4_2_0": "Climate solutions are ineffective",
    "claim_4_2_1": "Green economy won't work",
    "claim_4_2_10": "Future generations, technologies, and efficiencies will solve it",
    "claim_4_2_11": "Adaptation is the solution",
    "claim_4_2_12": "Energy efficiency is enough",
    "claim_4_2_14": "Other issues are more pressing",
    "claim_4_2_15": "Cheaper to mitigate abroad",
    "claim_4_2_2": "Policy impact is negligible",
    "claim_4_2_3": "One country is negligible",
    "claim_4_2_4": "Other countries' emissions",
    "claim_4_2_6": "Policies can be manipulated",
    "claim_4_2_7": "Climate-friendly alternatives are ineffective",
    "claim_4_2_8": "Markets are more efficient",
    "claim_4_2_9": "Individuals are responsible",
    "claim_4_3_0": "Solving climate change is too difficult",
    "claim_4_3_1": "We're not ready for policy",
    "claim_4_3_2": "It's too late to fix it",
    "claim_4_3_3": "Low support",
    "claim_4_4_0": "No need for more action",
    "claim_4_4_1": "Already taking it seriously",
    "claim_4_4_2": "Already doing enough",
    "claim_5_0_0": "Climate-related science is uncertain/unsound/unreliable (data, methods & models)",
    "claim_5_1_0": "There's no scientific consensus on climate/the science isn't settled",
    "claim_5_2_0": "Proxy data is unreliable (includes hockey stick)",
    "claim_5_3_0": "Temperature record is unreliable",
    "claim_5_4_0": "Models are wrong/unreliable/uncertain",
    "claim_6_0_0": "Proponents are alarmist",
    "claim_6_1_0": "Climate movement is alarmist (people or groups)",
    "claim_6_1_1": "Climate movement is religion",
    "claim_6_1_2": "Media (includes bloggers) is alarmist",
    "claim_6_1_3": "Politicians/government/UN are alarmist",
    "claim_6_1_4": "Environmentalists are alarmist",
    "claim_6_1_5": "Scientists/academics are alarmist",
    "claim_6_2_0": "Climate change (science or policy) is a conspiracy (deception)",
    "claim_7_0_0": "We need fossil fuels",
    "claim_7_1_0": "Fossil fuels are plentiful",
    "claim_7_2_0": "Fossil fuels are good",
    "claim_7_2_1": "Good for economic growth",
    "claim_7_2_2": "Good for energy security",
    "claim_7_2_3": "Our fossil fuels are clean",
    "claim_7_2_4": "Fossil fuels are part of the solution",
    "claim_7_3_0": "Fossil fuels are necessary",
    "claim_7_4_0": "We have the right to use them"
}

superclaim_map = {
    "claim_0": "No Claim detected",
    "claim_1": "Global warming is not happening",
    "claim_2": "Human greenhouse gases are not causing climate change",
    "claim_3": "Climate impacts/global warming is beneficial/not bad",
    "claim_4": "Climate solutions won't work",
    "claim_5": "Climate-related science is unreliable (data, methods & models)",
    "claim_6": "Proponents are alarmist or corrupt",
    "claim_7": "We need fossil fuels"
}

# Hard-coded subclaim_map for claims with exactly two underscores
subclaim_map = {
    'claim_0_0': 'No Claim detected',
    'claim_1_0': 'Global warming is not happening',
    'claim_1_1': "Ice/permafrost/snow cover isn't melting",
    'claim_1_2': "We're heading into an ice age/global cooling",
    'claim_1_3': 'Weather is cold/snowing',
    'claim_1_4': "Hasn't warmed over the last (few) decade(s)",
    'claim_1_5': 'Oceans are cooling/not warming',
    'claim_1_6': 'Sea level rise is exaggerated',
    'claim_1_7': "Extreme weather isn't linked to climate change",
    'claim_1_8': "They changed the name to 'climate change'",
    'claim_2_0': 'Human greenhouse gases are not causing climate change',
    'claim_2_1': "It's natural cycles",
    'claim_2_3': "No evidence for carbon dioxide driving climate",
    'claim_3_0': 'Global warming is beneficial (or not bad)',
    'claim_3_1': 'Climate sensitivity is low',
    'claim_3_2': "Species aren't showing climate impacts (or are benefiting)",
    'claim_3_3': 'CO2 is beneficial/not a pollutant',
    'claim_3_4': "It's only a few degrees (or less)",
    'claim_3_5': 'Climate change does not contribute to human conflict',
    'claim_3_6': "Climate change doesn't negatively impact health",
    'claim_4_0': "Climate solutions won't work",
    'claim_4_1': 'Climate solutions are harmful',
    'claim_4_2': 'Climate solutions are ineffective',
    'claim_4_3': 'Solving climate change is too difficult',
    'claim_4_4': 'No need for more action',
    'claim_5_0': 'Climate-related science is unreliable (data, methods & models)',
    'claim_5_1': "There's no scientific consensus",
    'claim_5_2': 'Proxy data is unreliable',
    'claim_5_3': 'Temperature record is unreliable',
    'claim_5_4': 'Models are wrong/unreliable',
    'claim_6_0': 'Proponents are alarmist',
    'claim_6_1': 'Proponents are alarmist (people or groups)',
    'claim_6_2': 'Climate change (science or policy) is a conspiracy (deception)',
    'claim_7_0': 'We need fossil fuels',
    'claim_7_1': 'Fossil fuels are plentiful',
    'claim_7_2': 'Fossil fuels are good',
    'claim_7_3': 'Fossil fuels are necessary',
    'claim_7_4': 'We have the right to use them'
}

# First, convert the date column to datetime if it's not already
df['date'] = pd.to_datetime(df['date'])

# Create a year column for aggregation
df['year'] = df['date'].dt.year

# For the bar chart (top claims as percentage of total)
# Count occurrences of each subclaim (excluding those ending in "_0")
subclaim_counts = {}
for subclaim in subclaim_map.keys():
    if subclaim in df.columns and not subclaim.endswith("_0"):  # Exclude subclaims ending in "_0"
        subclaim_counts[subclaim] = df[subclaim].sum()

# Convert to dataframe for easier manipulation
subclaim_df = pd.DataFrame({'claim': list(subclaim_counts.keys()), 
                           'count': list(subclaim_counts.values())})

# Add readable labels using subclaim_map
subclaim_df['label'] = subclaim_df['claim'].map(subclaim_map)

# Calculate proportion
total_subclaims = subclaim_df['count'].sum()
subclaim_df['proportion'] = subclaim_df['count'] / total_subclaims

# Sort and get top 15
top_subclaims = subclaim_df.sort_values('count', ascending=False).head(15)

# For the second bar chart - top subsubclaims (claims with 3 underscores, excluding those ending in "_0")
subsubclaim_counts = {}
subsubclaim_cols = [col for col in df.columns if col.startswith('claim_') and col.count('_') == 3 and not col.endswith('_0')]

for subsubclaim in subsubclaim_cols:
    if subsubclaim in df.columns:
        subsubclaim_counts[subsubclaim] = df[subsubclaim].sum()

# Convert to dataframe for easier manipulation
subsubclaim_df = pd.DataFrame({'claim': list(subsubclaim_counts.keys()), 
                              'count': list(subsubclaim_counts.values())})

# Add readable labels using subsubclaim_map
subsubclaim_df['label'] = subsubclaim_df['claim'].map(subsubclaim_map)

# Calculate proportion
total_subsubclaims = subsubclaim_df['count'].sum()
subsubclaim_df['proportion'] = subsubclaim_df['count'] / total_subsubclaims

# Sort and get top 15
top_subsubclaims = subsubclaim_df.sort_values('count', ascending=False).head(15)

# --- UPDATE: Use speech-level data instead of mention-level data ---
# Load the speech_data.csv file for speech-level aggregation
speech_df = pd.read_csv("data/speech_data.csv")

# Use claim_0 to claim_7 from speech data
speech_claim_cols = [f"claim_{i}" for i in range(8)]

# Group by year and sum claims for speech data
speech_claims_by_year = speech_df.groupby('year')[speech_claim_cols].sum()

# Calculate the total number of claims per year (across claim_0 to claim_7)
speech_total_claims_per_year = speech_claims_by_year.sum(axis=1)

# Calculate the proportion of each claim per year
speech_proportion_by_year = speech_claims_by_year.div(speech_total_claims_per_year, axis=0)

# Map speech-level claims to the visualization categories
yearly_data = pd.DataFrame(index=speech_proportion_by_year.index)
yearly_data['solutions_wont_work'] = speech_proportion_by_year['claim_4']
yearly_data['fossil_fuels_good'] = speech_proportion_by_year['claim_7'] 
yearly_data['science_unreliable'] = speech_proportion_by_year['claim_5']
yearly_data['proponents_alarmist'] = speech_proportion_by_year['claim_6']
yearly_data['not_happening'] = speech_proportion_by_year['claim_1']
yearly_data['not_us'] = speech_proportion_by_year['claim_2']
yearly_data['not_bad'] = speech_proportion_by_year['claim_3']

# Define the colors based on the image
color_map = {
    'claim_1': '#ef553bff',  # Red - Global warming isn't happening
    'claim_2': '#00cc96ff',  # Green - Humans are not causing climate change
    'claim_3': '#ab63faff',  # Purple - Climate change will be beneficial
    'claim_4': '#636efaff',  # Blue - Climate solutions won't work
    'claim_5': '#ff6692ff',  # Cyan - Climate science is unreliable
    'claim_6': '#19d3f3ff',  # Lavender - Proponents are alarmist
    'claim_7': '#ffa15aff'   # Orange - We need fossil fuels
}

# Define line styles and markers for additional differentiation
line_style_map = {
    'solid': '-',
    'dashed': '--', 
    'dotted': ':',
    'dashdot': '-.',
    'long-dash': (0, (8, 3))  # Customized long dash
}

marker_map = {
    'claim_1': 'x',     # X marker
    'claim_2': '^',     # Triangle marker
    'claim_3': '+',     # Plus marker
    'claim_4': 'o',     # Circle marker
    'claim_7': '^',     # Triangle marker
    'claim_5': 's',     # Square marker
    'claim_6': 'D',     # Diamond marker
}

# Convert linestyle names to actual matplotlib linestyle specifications
linestyle_convert = {
    'solid': '-',
    'dashed': '--', 
    'dotted': ':',
    'dashdot': '-.',
    'long-dash': (0, (8, 3))  # Customized long dash
}

# Create the visualization
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(6, 2, width_ratios=[1, 1.5], height_ratios=[1, 1, 1, 1, 1, 1])

# Bar chart (left top) - Top 15 Subclaims
ax0 = plt.subplot(gs[0:3, 0])  # Span first 3 rows (equal size)
top_subclaims_sorted = top_subclaims.sort_values('proportion')
colors = []
for claim in top_subclaims_sorted['claim']:
    # Get the first 7 characters (e.g., 'claim_1' from 'claim_1_1')
    claim_prefix = claim[:7]
    colors.append(color_map.get(claim_prefix, 'gray'))  # Default to gray if not found

ax0.barh(top_subclaims_sorted['label'], top_subclaims_sorted['proportion'], color=colors)
ax0.set_title('a) Top 15 Level 2 Claims (Proportion of Total Claims)', fontsize=14)
ax0.tick_params(axis='y', labelsize=11)  # Slightly bigger font for y-axis labels
ax0.tick_params(axis='x', labelsize=12)  # Add x-axis label size
# Adjust y-axis to reduce white space
ax0.margins(y=0.01)  # Reduce the y-margins to 1%

# Bar chart (left bottom) - Top 15 Subsubclaims  
ax4 = plt.subplot(gs[3:6, 0])  # Span last 3 rows (equal size)
top_subsubclaims_sorted = top_subsubclaims.sort_values('proportion')
colors_subsubclaims = []
for claim in top_subsubclaims_sorted['claim']:
    # Get the first 7 characters (e.g., 'claim_1' from 'claim_1_1_1')
    claim_prefix = claim[:7]
    colors_subsubclaims.append(color_map.get(claim_prefix, 'gray'))  # Default to gray if not found

ax4.barh(top_subsubclaims_sorted['label'], top_subsubclaims_sorted['proportion'], color=colors_subsubclaims)
ax4.set_xlabel('Proportion of Total Claims', fontsize=14)
ax4.set_title('b) Top 15 Level 3 Claims (Proportion of Total Claims)', fontsize=14)
ax4.tick_params(axis='y', labelsize=11)  # Slightly bigger font for y-axis labels
ax4.tick_params(axis='x', labelsize=12)  # Add x-axis label size
# Adjust y-axis to reduce white space
ax4.margins(y=0.01)  # Reduce the y-margins to 1%

# First time series (right top) - Solutions Won't Work & Fossil Fuels Are Good
ax1 = plt.subplot(gs[0:2, 1])  # First 2 rows (equal size)
ax1.plot(yearly_data.index, yearly_data['solutions_wont_work'], 
         marker=marker_map['claim_4'], linestyle=linestyle_convert['solid'], 
         color=color_map['claim_4'], linewidth=2, markersize=8, 
         label="Solutions won't work")
ax1.plot(yearly_data.index, yearly_data['fossil_fuels_good'], 
         marker=marker_map['claim_7'], linestyle=linestyle_convert['dashed'], 
         color=color_map['claim_7'], linewidth=2, markersize=8, 
         label="Fossil fuels are good")
ax1.set_title('c) Solutions Won\'t Work & Fossil Fuels are Good', fontsize=14)
ax1.legend(fontsize=12)
ax1.set_xlim(1994, 2024)
ax1.set_ylabel('', fontsize=12)
ax1.tick_params(axis='both', labelsize=12)

# Second time series (right middle) - Science is Unreliable & Proponents are Alarmist 
ax2 = plt.subplot(gs[2:4, 1])  # Middle 2 rows (equal size)
ax2.plot(yearly_data.index, yearly_data['science_unreliable'], 
         marker=marker_map['claim_5'], linestyle=linestyle_convert['solid'], 
         color=color_map['claim_5'], linewidth=2, markersize=8, 
         label="Science is unreliable")
ax2.plot(yearly_data.index, yearly_data['proponents_alarmist'], 
         marker=marker_map['claim_6'], linestyle=linestyle_convert['dashed'], 
         color=color_map['claim_6'], linewidth=2, markersize=8, 
         label="Proponents are alarmist")
ax2.set_title('d) Science is Unreliable & Proponents are Alarmist', fontsize=14)
ax2.legend(fontsize=12)
ax2.set_ylabel('Proportion of speeches', fontsize=14)
ax2.set_xlim(1994, 2024)
ax2.tick_params(axis='both', labelsize=12)

# Third time series (right bottom) - It's Not Happening, It's Not Us, It Won't Be Bad
ax3 = plt.subplot(gs[4:6, 1])  # Last 2 rows (equal size)
ax3.plot(yearly_data.index, yearly_data['not_happening'], 
         marker=marker_map['claim_1'], linestyle=linestyle_convert['solid'], 
         color=color_map['claim_1'], linewidth=2, markersize=8, 
         label="It's not happening")
ax3.plot(yearly_data.index, yearly_data['not_us'], 
         marker=marker_map['claim_2'], linestyle=linestyle_convert['dashed'], 
         color=color_map['claim_2'], linewidth=2, markersize=8, 
         label="It's not us")
ax3.plot(yearly_data.index, yearly_data['not_bad'], 
         marker=marker_map['claim_3'], linestyle=linestyle_convert['dotted'], 
         color=color_map['claim_3'], linewidth=2, markersize=8, 
         label="It won't be bad")
ax3.set_title('e) It\'s Not Happening, It\'s Not Us, It Won\'t Be Bad', fontsize=14)
ax3.legend(fontsize=12)
ax3.set_xlabel('Year', fontsize=14)
ax3.set_ylabel('', fontsize=12)
ax3.set_xlim(1994, 2024)
ax3.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('replicate_statistical_analysis/figure2.pdf', bbox_inches='tight')
plt.show()

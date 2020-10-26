# Determine the 5 biggest county case rates in these 5 states:
def get_county_state_dict(path):
    top5_df = pd.read_excel(path)

    state_list = top5_df['State'].unique().tolist()

    county_dict = {}
    for state in state_list:
        county_dict[state] = top5_df[top5_df['State'] == state]['Counties'].tolist()
    return pd.DataFrame.from_dict(county_dict)


def get_population_dict(path):
    df = pd.read_excel(path, skiprows=2, skipfooter=5)
    new_df = df[['Geographic Area', 'Unnamed: 12']].reset_index().iloc[1:].reset_index()
    Area_list = new_df['Geographic Area']
    area_list = [i.split(',')[0].split(' ')[0].replace('.', '') for i in Area_list]
    new_df['Geographic Area'] = area_list
    return new_df


state_name  = 'Texas'
county_df =  get_county_state_dict(path= 'Top5counties.xlsx')

pop_df = get_population_dict("https://www2.census.gov/programs-surveys/popest/tables/2010-2019/counties/totals/co-est2019-annres-48.xlsx")

# NY, NJ, CA, MI, PA, TX
def get_county_pop_list(county_df, pop_df):
    county_list = county_df[state_name].tolist()
    new_pop_df = pop_df[pop_df['Geographic Area'].isin(county_list)]
    pop_list = new_pop_df['Unnamed: 12'].tolist()
    return county_list, pop_list

county_list, pop_list = get_county_pop_list(county_df, pop_df)
counties=[]
hospital_beds = [19000,14000,5000,5000,7893]
for i in range(len(county_list)):
    counties.append([county_list[i], state_name, pop_list[i], hospital_beds[i]])

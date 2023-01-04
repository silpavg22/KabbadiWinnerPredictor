import pandas as pand
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


kabbadi_teams = ['Telugu Titans', 'U Mumba', 'Patna Pirates', 'Bengaluru Bulls',
            'Fortune Giants', 'Dabang Delhi', 'Tamil Thalaivas', 'Puneri Paltan',
            'Bengal Warriors', 'UP Yoddha', 'Pink Panthers', 'Haryana Steelers']

file_path='./PKB.xlsx'
scores = pand.read_excel(file_path)

def find_totalteams_removeduplicates(scores,kabbadi_teams):

    winningteam = []
    for i in range (len(scores['FirstTeam'])):
        if scores ['Score_FirstTeam'][i] > scores['Score_SecondTeam'][i]: #comparing the scores and adding winner to winningteam
            winningteam.append(scores['FirstTeam'][i])
        elif scores['Score_FirstTeam'][i] < scores ['Score_SecondTeam'][i]:
            winningteam.append(scores['SecondTeam'][i])
        else:
            winningteam.append('Tie')
    scores['team_won'] = winningteam #appending the winner of each match in the scores array

    teams1 = scores[scores['FirstTeam'].isin(kabbadi_teams)] #teams in the first column of dataset
    teams2 = scores[scores['SecondTeam'].isin(kabbadi_teams)] #teams in the second column of dataset
    total_teams = pand.concat((teams1, teams2))
    total_teams.drop_duplicates() # dropping the duplicate teams
    total_teams = total_teams.drop(['Time', 'Score_FirstTeam', 'Score_SecondTeam', 'WinnerTeam', 'ScoreDifference'],
                                   axis=1)  # removing the columns that does not affect the prediction
    total_teams.loc[total_teams.team_won == total_teams.FirstTeam, 'team_won'] = 0  # If the winner is first team then add '0' for team_won column, '1' for second team else it is a tie
    total_teams.loc[total_teams.team_won == 'Tie', 'team_won'] = 1
    total_teams.loc[total_teams.team_won == total_teams.SecondTeam, 'team_won'] = 2
    return total_teams;


total_teams=find_totalteams_removeduplicates(scores,kabbadi_teams);
final = pand.get_dummies(total_teams, prefix=['FirstTeam', 'SecondTeam'], columns=['FirstTeam', 'SecondTeam'])
X = final.drop(['team_won'], axis=1)#all the data except team_won is taken as X and team_won colum is taken as Y
Y = final["team_won"]
Y = Y.astype('int')


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
logregn = LogisticRegression()
logregn.fit(X_train, y_train)
Y_pred = logregn.predict(X_test)

print("\nAccuracy of Training Set: ", '%.3f'%(logregn.score(X_train, y_train)))
print("Accuracy of Test Set: ", '%.3f'%(logregn.score(X_test, y_test)))

ranking = pand.read_excel(file_path,'Points')#read the dataset containing win points of all teams

score_fixtures = pand.read_csv('./score_fixtures.csv')#read the dataset containing matches for which prediction is to be done
score_fixtures = score_fixtures.iloc[:131, :] #removing the matches that is to be announced


def find_predictionset(score_fixtures,ranking):
    prediction_set = []
    score_fixtures.insert(1, 'First_Position', score_fixtures['Team_1'].map(ranking.set_index('Team')['WINS']))#check whether the teams have wins and add to score_fixtures
    score_fixtures.insert(2, 'Second_Position', score_fixtures['Team_2'].map(ranking.set_index('Team')['WINS']))
    for index, row in score_fixtures.iterrows():
        if row['First_Position'] > row['Second_Position']: #according to the 'Wins' placing the team with higher Wins in FirstPosition
            prediction_set.append({'FirstTeam': row['Team_1'], 'SecondTeam': row['Team_2'], 'team_won': None})
        else:
            prediction_set.append({'FirstTeam': row['Team_2'], 'SecondTeam': row['Team_1'], 'team_won': None})

    prediction_set = pand.DataFrame(prediction_set) #converting array into dataframe
    return prediction_set


prediction_set = find_predictionset(score_fixtures,ranking)
previous_prediction_set = prediction_set #storing in another backup variable
prediction_set = pand.get_dummies(prediction_set, prefix=['FirstTeam', 'SecondTeam'], columns=['FirstTeam', 'SecondTeam'])
#converting categorical data to continuous input values to apply logistic regression alogorithm


def find_missingcolumns_append0(prediction_set):
    missing_columns = set(final.columns) - set(prediction_set.columns)#finding the missing colums by comparing with finaldata
    for c in missing_columns:
        prediction_set[c] = 0 #appending 0 to missing columns to match the columns
    prediction_set = prediction_set[final.columns]
    prediction_set = prediction_set.drop(['team_won'], axis=1)
    return prediction_set


prediction_set = find_missingcolumns_append0(prediction_set)
predicted_values = logregn.predict(prediction_set) #predicting the winner
for i in range(score_fixtures.shape[0]):
    print("Game " + str(i+1) + ": "+previous_prediction_set.iloc[i, 1] + " and " + previous_prediction_set.iloc[i, 0])
    if predicted_values[i] == 0:  #0 then team in the FirstPoistion won
        print("Winner: " + previous_prediction_set.iloc[i, 0])
    elif predicted_values[i] == 2:  #2 then team in the SecondPosition won
        print("Winner: " + previous_prediction_set.iloc[i, 1])
    else:
        print("Tie between two teams")  # else there is draw between teams
    print("")


def cleaning_prediction(teams, level,ranking, final, logreg): # function for model predicting the winner by providing two teams as input using the learned data
    positions = []
    for team in teams:
        positions.append(ranking.loc[ranking['Team'] == team[0],'WINS'].iloc[0])
        positions.append(ranking.loc[ranking['Team'] == team[1],'WINS'].iloc[0])

    prediction_set = []
    i = 0
    while i < len(positions):
        dict_one = {}

        if positions[i] > positions[i + 1]: #comparing thw Wins of teams and adding as FirstTeam and SecondTeam respectively
            dict_one.update({'FirstTeam': team[0], 'SecondTeam': team[1]})
        else:
            dict_one.update({'FirstTeam': team[1], 'SecondTeam': team[0]})

        prediction_set.append(dict_one)
        i += 2

    prediction_set = pand.DataFrame(prediction_set)
    backup_prediction_set = prediction_set
    prediction_set = pand.get_dummies(prediction_set, prefix=['FirstTeam', 'SecondTeam'], columns=['FirstTeam', 'SecondTeam'])

    missing_cols2 = set(final.columns) - set(prediction_set.columns)
    for c in missing_cols2:
        prediction_set[c] = 0
    prediction_set = prediction_set[final.columns]
    prediction_set = prediction_set.drop(['team_won'], axis=1)

    predictions = logreg.predict(prediction_set)
    for i in range(len(prediction_set)):
        print(backup_prediction_set.iloc[i, 1] + " and " + backup_prediction_set.iloc[i, 0])
        if predictions[i] == 0:
            print("Winner of "+level+": " + backup_prediction_set.iloc[i, 0])
        elif predictions[i] == 2:
            print("Winner of "+level+": " + backup_prediction_set.iloc[i, 1])
        print("")

print('Predicting the winner of two games:-\n')
game1 = [('Haryana Steelers', 'Bengaluru Bulls')]
game2 = [('U Mumba', 'UP Yoddha')]
cleaning_prediction(game1, 'Game 1',ranking, final, logregn)
cleaning_prediction(game2, 'Game 2',ranking, final, logregn)#calling prediction function to predict winner for game1 and game2

plt.title('Winning chances of teams')
plt.bar(ranking.Team,ranking.WINS,facecolor='green', alpha=0.5)
plt.xticks(rotation=45)
plt.show()











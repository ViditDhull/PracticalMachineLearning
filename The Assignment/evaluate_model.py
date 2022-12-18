from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

def evaluate_model(model, X_test, y_test):
	'''Predicts and evaluates the model'''
	
	predictions = model.predict(X_test)

	accuracy = accuracy_score(y_test, predictions)
	precision = precision_score(y_test, predictions)
	recall = recall_score(y_test, predictions)
	f1 = f1_score(y_test, predictions)
	tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

	return accuracy, precision, recall, f1, fp
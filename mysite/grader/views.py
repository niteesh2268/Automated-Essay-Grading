from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse

from .models import Question, Essay
from .forms import AnswerForm

from .utils.model import *
from .utils.helpers import *
from .utils.help import *

import os
import dill

current_path = os.path.abspath(os.path.dirname(__file__))

# Create your views here.
def index(request):
    questions_list = Question.objects.order_by('set')
    context = {
        'questions_list': questions_list,
    }
    return render(request, 'grader/index.html', context)

def essay(request, question_id, essay_id):
    essay = get_object_or_404(Essay, pk=essay_id)
    context = {
        "essay": essay,
    }
    return render(request, 'grader/essay.html', context)


def question(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = AnswerForm(request.POST)
        if form.is_valid():

            content = form.cleaned_data.get('answer')
            '''
            if len(content) > 20:
                num_features = 300
                model = word2vec.KeyedVectors.load_word2vec_format(os.path.join(current_path, "deep_learning_files/word2vec.bin"), binary=True)
                clean_test_essays = []
                clean_test_essays.append(essay_to_wordlist( content, remove_stopwords=True ))
                testDataVecs = getAvgFeatureVecs( clean_test_essays, model, num_features )
                testDataVecs = np.array(testDataVecs)
                testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

                lstm_model = get_model()
                lstm_model.load_weights(os.path.join(current_path, "deep_learning_files/final_lstm.h5"))
                preds = lstm_model.predict(testDataVecs)

                if math.isnan(preds):
                    preds = 0
                else:
                    preds = np.around(preds)

                if preds < 0:
                    preds = 0
                if preds > question.max_score:
                    preds = question.max_score
            else:
                preds = 0
            '''
######################################################################################

            # train_data = pd.read_csv(os.path.join(current_path,"ml_files/training_set_rel3.tsv"), delimiter='\t', encoding='ISO-8859-1')
            # train_data = train_data[['essay_id', 'essay_set', 'essay', 'rater1_domain1', 'rater2_domain1', 'rater1_domain2', 'rater2_domain2', 'domain1_score', 'domain2_score']]
            # train_data['score'] = train_data['domain1_score']
            # features_set = pickle.load(open(os.path.join(current_path,"ml_files/features.sav"), 'rb'))
            # y = np.array(train_data['score'], dtype = int)
            # X = features_set.iloc[:, 3:]
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
            feature_list = ['Sentence Count', 'Word Count', 'Bigram Count', 'Noun Count', 'Adjective Count', 'Verb Count', 'Adverb Count']
            # X_train_np = X_train.to_numpy()
            # explainer = lime.lime_tabular.LimeTabularExplainer(training_data = X_train_np, training_labels=y_train, feature_names=feature_list, verbose=True, mode='regression')
            explainer = dill.load(open(os.path.join(current_path,"ml_files/explainer.sav"), 'rb'))
            if len(content) > 20:
            	essay_features = pd.DataFrame([content], columns = ['essay'])
            	essay_features = extract_features(essay_features)
            	rfr_model = pickle.load(open(os.path.join(current_path,"ml_files/rfr_100.sav"), 'rb'))
            	preds = rfr_model.predict(essay_features.iloc[:, 1:])
            	test_features = essay_features.iloc[:, 1:].to_numpy()
            	exp = explainer.explain_instance(data_row = test_features[0], predict_fn = rfr_model.predict, num_features=7)
            	exp.show_in_notebook(show_table=True)
            	exp.save_to_file('grader/static/lime1.html')
            	f_importances(rfr_model.feature_importances_, feature_list)
            	plt1 = exp.as_pyplot_figure()
            	buf = BytesIO()
            	plt.figure(figsize=(8,7))
            	#exp.save_to_file('fig.png')
            	plt1.savefig(buf, format='png', dpi=300)
            	image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
            	buf.close()
		
            	if math.isnan(preds):
                	preds = 0
            	else:
                	preds = np.around(preds)

            	if preds < 0:
               		preds = 0
            	if preds > question.max_score:
                	preds = question.max_score
            else:
            	preds = 0

################################################################################
            # K.clear_session()
            essay = Essay.objects.create(
                content=content,
                question=question,
                score=preds
            )
        return redirect('essay', question_id=question.set, essay_id=essay.id)
    else:
        form = AnswerForm()

    context = {
        "question": question,
        "form": form,
    }
    return render(request, 'grader/question.html', context)

from django.shortcuts import render, redirect

# Create your views here.
# pip install Pillow
# pip install numpy
# pip install google.cloud
# pip install google-cloud-vision
# !pip install google-cloud-language
# pip install --upgrade google-cloud-storage
# pip install pandas
# pip install google-cloud-translate==3.0.2
# pip install --upgrade google-cloud-texttospeech
# pip3 install lxml
# pip3 install bs4

import io, os
from numpy import random
from django.http.response import HttpResponse
from django.http import HttpResponseRedirect
from google.cloud import vision_v1, translate_v2, language_v1#,texttospeech_v1
from PIL import Image
# from Pillow_Utility import draw_borders, Image
import pandas as pd
from .forms import LanguageForm, RecordForm
from .models import Languages, Record, Texts
import six
from json import dumps
import re 
from random import randint
import requests
import string
import urllib.request
from bs4 import BeautifulSoup


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/emilyhuang/Downloads/service-account-file.json"
client = vision_v1.ImageAnnotatorClient()

syntax_dict = {
    'UNKNOWN':'unknown',
    'ADJ':'adjective',
    'ADP':'adposition',
    'ADV':'adverb',
    'CONJ':'conjunction',
    'DET':'determiner',
    'NOUN': 'noun',
    'NUM': 'number',
    'PRON': 'pronoun',
    'PRT': 'particle',
    'PUNCT':'punctuation',
    'VERB': 'verb',
    'X':'abbrev',
    'AFFIX':'affix',

}
def sample_analyze_syntax(text_content):
    """
    Analyzing Syntax in a String

    Args:
      text_content The text content to analyze
    """

    client = language_v1.LanguageServiceClient()

    # text_content = 'This is a short sentence.'

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_syntax(request = {'document': document, 'encoding_type': encoding_type})
    # Loop through tokens returned from the API
    for token in response.tokens:
        # Get the text content of this token. Usually a word or punctuation.
        text = token.text
        print(u"Token text: {}".format(text.content))
        print(
            u"Location of this token in overall document: {}".format(text.begin_offset)
        )
        # Get the part of speech information for this token.
        # Parts of spech are as defined in:
        # http://www.lrec-conf.org/proceedings/lrec2012/pdf/274_Paper.pdf
        part_of_speech = token.part_of_speech
        # Get the tag, e.g. NOUN, ADJ for Adjective, et al.
        print(
            u"Part of Speech tag: {}".format(
                language_v1.PartOfSpeech.Tag(part_of_speech.tag).name
            )
        )
        tag_name = language_v1.PartOfSpeech.Tag(part_of_speech.tag).name
        part_of_speech1 = syntax_dict[tag_name]
        # print(part_of_speech1)
        return part_of_speech1

# sample_analyze_syntax('rimlands')

# other needed functions
def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    # import six
    # from google.cloud import translate_v2 as translate

    translate_client = translate_v2.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    print(u"Text: {}".format(result["input"]))
    print(u"Translation: {}".format(result["translatedText"]))
    print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    return result

def detect_document(path):
    """Detects document features in an image."""
    client = vision_v1.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision_v1.Image(content=content)

    response = client.document_text_detection(image=image)
    """
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(
                    paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))

                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))
"""
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


# contains elements from all scans
total = set()#[]
translated = []
input_output = {}

# views
def detect_view(request):
    # contains elements only from the current scan
    current = []
    if request.method == 'POST': 
        path = request.FILES['myfile']
        content = path.read()
        image = vision_v1.types.Image(content =content)
        response = client.object_localization(image=image)
        localized_object_annotations = response.localized_object_annotations
        # pillow_image = Image.open(image_path)
        # df = pd.DataFrame(columns=['name', 'score'])
        for obj in localized_object_annotations:
            current.append(obj.name)
            total.add(obj.name)
            print(obj.name)
        print("current: ", current)
        print("total: ", total)
        
        return redirect('/language')
    context = {
            'names':current,
            'total': total
    }
    return render(request,"detect/detect_detail.html", context)

def language_view(request):
    form = LanguageForm(request.POST or None)
    if form.is_valid():
        form.save()
        form = LanguageForm()
        return redirect('/flashcards#home')
    context = {
        'form': form
    }
    return render(request, "detect/language_select.html", context)

def card_view(request):
    language_list = Languages.objects.all()
    # equivalent to the current language code
    lang_code = language_list.latest('id').language
    
    # equivalent to the previous language code
    reverse = language_list.order_by('-language')
    lang_code_prev = reverse[1].language
    """
    if lang_code != lang_code_prev:
        translated.clear()
    """
    translated.clear()
    for word in total:
        translated_word = translate_text(lang_code, word)["translatedText"]
        translated.append(translated_word)
        # maps word to translated word
        input_output[word] = translated_word
    
    dataJSON = dumps(input_output)

    # michelle's
    question_dict = {} #dict accessed in activity.json
    question_list = [] #will be list containing dictionaries 
    question_dict["questionlist"] = question_list # question_dict has 1 key

    for word in total: 
        word_dict = {} 
        word_dict["cardfront"] = word #original 
        word_dict["cardback"] = translate_text(lang_code,word)["translatedText"] #translated
        question_list.append(word_dict) #add dictionary to list

    # end of michelle's 


    question_dict = dumps(question_dict)
    context = {
        'translated': translated,
        'translate_dict': dataJSON,
        'input_output': input_output,
        'question_dict': question_dict
        
    }
    print("translated: ", translated)
    print("total: ", total)
    print("input_output", input_output)

    return render(request, "detect/card_detail.html",context)

# pillow_image.show()


def document_view(request):
    if request.method == 'POST': 
        path = request.FILES['afile']
        content = path.read()
        image = vision_v1.types.Image(content =content)
        response = client.document_text_detection(image=image)

        texts = response.text_annotations
        print(texts)
        """
        for text in texts:
            print('\n"{}"'.format(text.description))
        """

        """
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                print('\nBlock confidence: {}\n'.format(block.confidence))

                for paragraph in block.paragraphs:
                    print('Paragraph confidence: {}'.format(
                        paragraph.confidence))

                    for word in paragraph.words:
                        word_text = ''.join([
                            symbol.text for symbol in word.symbols
                        ])
                        print('Word text: {} (confidence: {})'.format(
                            word_text, word.confidence))

                        for symbol in word.symbols:
                            print('\tSymbol: {} (confidence: {})'.format(
                                symbol.text, symbol.confidence))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))
        """
    return render(request,"detect/document_detail.html", {})

def recording_view(request):
    text_list = Texts.objects.all()
    num_texts = Texts.objects.count()

    random_object = Texts.objects.all()[randint(0, num_texts - 1)]

    text = random_object.passage
    # text = text_list.latest('id').passage
    request.session['text'] = text
    #Texts.objects.filter(level =)
    
    form = RecordForm(request.POST or None)
    if form.is_valid():
        form.save()
        form = RecordForm()
        return redirect('/results')
    context = {
        'form': form,
        'text': text
    }
    return render(request, "detect/recording_create.html", context)

# compare the two texts and identify a score
def gradeTranscript1(original, transcript):
    """
    Note: dictionaries are ordered python 3.6+
    Compare the original to the transcript and assign an accuracy score.
    Input:
        - original: a string of sentences representing the original text
        - transcript: a string of sentences representing the spoken transcript
    Ouptut:
        - score: an integer from 0-100 graded on the accuracy of the transcript
    """
    score = 100
    error = 0
    total_words = 0

    original_dict = {}
    transcript_dict = {}
    transcript1 = transcript 
    original_list = createWordList(original)
    transcript_list = createWordList(transcript1)

    # create dictionary mapping word to occurences for original text
    for sentence in original_list:  # "Cat jumps over dog.""
        sentence = sentence[
            :-1
        ]  # "Cat jumps over dog" aka removes end punctation like . ? !
        words = sentence.lower().split(" ")  # ["cat", "jumps", "over", "dog"]
        for word in words:
            if word in original_dict:
                original_dict[word] += 1
            else:
                original_dict[word] = 1
            total_words += 1

    # create dictionary mapping word to occurences for transcript text
    for sentence in transcript_list:  # "Cat jumps over dog.""
        sentence = sentence[
            :-1
        ]  # "Cat jumps over dog" aka removes end punctation like . ?
        words = sentence.lower().split(" ")  # ["cat", "jumps", "over", "dog"]
        for word in words:
            if word in transcript_dict:
                transcript_dict[word] += 1
            else:
                transcript_dict[word] = 1

    # check for words in original_dict
    for word in original_dict:
        if word in transcript_dict.keys():
            error += abs(original_dict[word] - transcript_dict[word])
        else:
            error += 1

    score = (1 - error / total_words) * 100
    return round(score)

def gradeTranscript(original, transcript):
   """
   Note: dictionaries are ordered python 3.6+
   Compare the original to the transcript and assign an accuracy score.
   Input:
       - original: a string representing the original text
       - transcript: a string representing the spoken transcript
   Ouptut:
       - score: an integer from 0-100 graded on the accuracy of the transcript
   """
   score = 100
   error = 0
   total_words = 0

   original_list = createWordList(original)
   transcript_list = createWordList(transcript)

   original_dict = createWordDict(original_list)[0] # first return value in createWordDict is dict
   transcript_dict = createWordDict(transcript_list)[0]
   total_words = createWordDict(original_list)[1] # second return value in createWordDict is word count

   # check for words in original_dict
   for word in original_dict:
       if word in transcript_dict.keys():
           error += abs(original_dict[word] - transcript_dict[word])
       else:
           error += 1

   score = (1 - error / total_words) * 100
   return score


def createWordList1(text):
    """
    Input:
        - text: a string list of sentences
    Ouptut:
        - sentence_list: list with each sentence of text as element, including periods
    """
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

    ###### score
def createWordList(text):
    """
    Inputs : text
    Outputs: list of words 
    """
    delimiters = "“", "-", "; ", ": ", "?", "!", ",", " ", "\n", "--"
    regexPattern = '|'.join(map(re.escape, delimiters))
    temp_list = re.split(regexPattern, text)
    word_list = []
    # cleaning up words so we can compare later

    for i in range(len(temp_list)):
        # remove punctuation and set to lowercase
        temp_list[i] = temp_list[i].translate(str.maketrans('', '', string.punctuation)).lower()
        temp_list[i] = temp_list[i].replace('“', '')  # remove quotation marks

    for word in temp_list:
        if word != '':
            word_list.append(word)
    return word_list
def createWordDict(text_list):
    """
    Input:
        text_list : a list with words as elements from a body of text

    Output:
        word_dict : dict mapping words to sentences in order they appear
    """
    word_dict = {}
    total_words = 0 # represents unique word count
    for word in text_list:  # "Cat jumps over dog.""
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1
            total_words += 1 # only counts # of unique words
    return word_dict, total_words
def definition(word):
    url = "https://www.vocabulary.com/dictionary/" + word + ""
    htmlfile = urllib.request.urlopen(url)
    soup = BeautifulSoup(htmlfile, 'html.parser')

    #soup1 = soup.find(class_="short")
    soup1 = soup.find(class_="definition")

    try:
        soup1 = soup1.get_text()
    except AttributeError:
        # print('Cannot find such word! Check spelling.')
        #exit()

    # Print short meaning
    # print ('-' * 25 + '->',word,"<-" + "-" * 25)
    # print ("SHORT MEANING: \n\n",soup1)
    # print ('-' * 65)
        """
    # Print long meaning
    try:
        soup2 = soup.find(class_="long")
        soup2 = soup2.get_text()
    
        print ("LONG MEANING: \n\n",soup2)

        print ('-' * 65)
    except:
        print("no long definition")   

    # Print instances like Synonyms, Antonyms, etc.
    try:
        soup3 = soup.find(class_="instances") 
        txt = soup3.get_text()
        txt1 = txt.rstrip()
        print (' '.join(txt1.split()))    
    except: 
        print("no additional instances")
    
        """

    
    # removing extra spaces and tabs
    soup_list = soup1.split()
    # part = " ".join(soup_list[0])

    soup_string = " ".join(soup_list[1:])

    # to remove everything after the semicolon in a definition
    simplified_list = soup_string.split(";",1)
    simplified = simplified_list[0]

    # returns the short meaning, part of speech
    return simplified# , part

def identifyError1(original_1, transcript_1):
    """
    Identifies which words in a transcript do not match up with the original.
    Input:
        - original: a string list with a sentence for each element of the original text
        - transcript: a string list with a sentence for each element of the spoken transcript
    Ouptut:
        - error_list: a list of words that are in the original but not transcript
    """
    original_dict = {}
    transcript_dict = {}
    error_list = []

    transcript_list = createWordList(transcript_1)
    print("transcript_list in identifyError:\n ", transcript_list)
    original_list = createWordList(original_1)
    print("original_list in identifyError: \n", original_list)

    # create dictionary mapping word to occurences for original text
    for sentence in original_list:  # "Cat jumps over dog.""
        sentence = sentence[
            :-1
        ]  # "Cat jumps over dog" aka removes end punctation like . ?
        words = sentence.lower().split(" ")  # ["cat", "jumps", "over", "dog"]
        for word in words:
            if word in original_dict:
                original_dict[word] += 1
            else:
                original_dict[word] = 1

    # create dictionary mapping word to occurences for transcript text
    for sentence in transcript_list:  # "Cat jumps over dog.""
        sentence = sentence[
            :-1
        ]  # "Cat jumps over dog" aka removes end punctation like . ?
        words = sentence.lower().split(" ")  # ["cat", "jumps", "over", "dog"]
        for word in words:
            if word in transcript_dict:
                transcript_dict[word] += 1
            else:
                transcript_dict[word] = 1

    for word in original_dict:
        if word not in transcript_dict.keys():  # if word not in transcript
            if word[-1] == "," or word[-1] == ".":
                error_list.append(word[0:-1])
            else:
                error_list.append(word)

    return error_list


def identifyError(original, transcript):
   """
   Identifies which words in a transcript do not match up with the original.
   Input:
       - original: a string list with a sentence for each element of the original text
       - transcript: a string list with a sentence for each element of the spoken transcript
   Ouptut:
       - error_list: a list of words that are in the original but not transcript
   """

   error_list = []

   original_list = createWordList(original)
   transcript_list = createWordList(transcript)

   #original_dict = createWordDict(original_list)[0]  # first return value in createWordDict is dict
   #transcript_dict = createWordDict(transcript_list)[0]

   total_words = createWordDict(original_list)[1]

   for i in range(total_words): # if word not in transcript
           if original_list[i] != transcript_list[i]:
               error_list.append(original_list[i])

   return error_list


"""
def yt_search(error_list):
    pronunc = {}
    search_url = 'https://www.googleapis.com/youtube/v3/search'
    video_url = 'https://www.googleapis.com/youtube/v3/videos'
    
    for error in error_list:
        keyword = str(error) + " pronunciation"

        search_params = {
            'part': 'snippet',
            'q': keyword,
            'key': 'AIzaSyDzHsY1o1ySzENf7O4dm7wR47wD-fhk5PQ',
            'maxResults': 1,
            'type': 'video'
        }

        video_ids = []
        r = requests.get(search_url, params=search_params)
        results = r.json()['items']
        for result in results:
            video_ids.append(result['id']['videoId'])

        video_params = {
            'part': 'snippet,contentDetails',
            'key': 'AIzaSyDzHsY1o1ySzENf7O4dm7wR47wD-fhk5PQ',
            'id': ','.join(video_ids),
            'max_results': 1
        }
        
        r = requests.get(video_url, params=video_params)
        results = r.json()['items']
        videos = []
        for result in results:
            # print(result['contentDetails']['duration'])
            url = f'https://www.youtube.com/watch?v={result["id"]}'
        
        pronunc[error] = url

    return pronunc
"""
def analyzeFrequencyList(word_list):
    """
    Given a string, return a dictionary mapping each word to its frequency (percentile) in langauge use
    Inputs:
        word_list : a string of words (paragraph usua)ly
    Outputs:
        analysis_dict : word mapped to frequency percentile 
    """
    frequency_list = []
    analysis_dict = {}
    
    with open("/Users/emilyhuang/Feb13/Education/src/detect/commonwords.txt") as file:
        for line in file:
            line_list = line.split("\t")
            # make dictionary of frequency_list
            word = line_list[0].lower()  # lowercase
            # frequency = float(line_list[1])  # the higher the number, the more frequent
            frequency_list.append(word)  # add word to list in order of frequency

    total_words = len(frequency_list)

    wordrank = 0

    for word in word_list:
        if word not in analysis_dict.keys():
            if word in frequency_list:
                wordrank = (
                    round(1 - (frequency_list.index(word) / total_words), 2) * 100
                )  # percentile
        analysis_dict[word] = wordrank

    return analysis_dict

def error_map(terms, actual_terms):
    """
    terms: a list of separate words
    error_list: a list of error words

    output: map words to 0 or 1
    0 means it is an error
    1 means it is not an error
    """
    error_mapping = {}
    part = " ".join(terms)
    # removes periods
    part = part.replace(".","")
    part = part.lower()
    term_list = part.split()

    part1 = " ".join(actual_terms)
    # removes periods
    part1 = part1.replace(".","")
    part1 = part1.lower()
    actual_list = part1.split()


    for term in term_list:
        if term in actual_list:
            error_mapping[term] = "1"
        else:    
            error_mapping[term] = "0"
    return error_mapping         

def result_view(request):

    # original text
    text = request.session['text']
    record_list = Record.objects.all()
    language_list = Languages.objects.all()

    # user-input recording
    recording_last = record_list.latest('id').recording
    # print("recording_last: \n", recording_last)
    # print("\n")
    space_sep = recording_last.split()
    
    originalText = str(text)
    # print("originalText\n", originalText)
    # By the time he reached the first talus slides under the tall escarpments, the dawn was not far. He brought the horses in a grassy swale. Coyotes were yapping along the hills on the rimlands above him.
    
    actual_sep = originalText.split()
    # transcript with mistakes
    # The weather was rice today so I went out on a ran. Tomorrow a storm is coming so I will stay. Hoping school gets canceled.
    # transcriptText = "by the time he reached the first Tails slides under the tallest statements the Dawn was not far he bought the horses in a grassy Swale cats were yapping along the hills on the rims above him"
    # transcriptText = "by the time he. "
    transcriptText = str(recording_last) + "."
    # transcriptText  = "I went to the mall."
    # print("transcriptText\n", transcriptText)
    # by the time he reached the first Tails slides under the tallest statements the Dawn was not far he bought the horses and a grassy swell cats were yapping along the hills on the rims above him
   
   
    # transcript without mistakes
    #transcriptText2 = """The weather was nice today so I went out on a run. Tomorrow a storm is coming so I will stay inside. Hopefully school gets canceled."""

    score_user = gradeTranscript(originalText, transcriptText)

    # returns a list of corrected words
    errors = identifyError(originalText,transcriptText)
    """
    errors = ['by', 'the', 'time', 'he', 'reached', 'first', 'talus', 'slides', 'under', 'tall', 'escarpments,', 'dawn', 'was', 'not', 'far', 
    'brought', 'horses', 'in', 'a', 'grassy', 'swale', 'coyotes', 'were', 'yapping', 
    'along', 'hills', 'on', 'rimlands', 'above', 'him']
    """
    # print("errors: ",errors)
    # returns a list of wrong words
    errors1 = identifyError(transcriptText, originalText)
    #print(errors1)

    #maps errors ot yt links
    card_dict = {}#yt_search(errors)

    # dictionary that maps each word in errors list to its common score
    frequency = analyzeFrequencyList(errors)


    # error check mapping word to 0 or 1, check whether its an error or not
    mapping = error_map(space_sep,actual_sep)
    
    #print("mapping", mapping)
    request.session['score'] = score_user

    # maps each word in errors to definition
    definitions = {}
    lang_code = language_list.latest('id').language
    for error in errors:
        # print("error: ",error)
        part_speech = sample_analyze_syntax(error)
        word_def = definition(error)
        new_translate = translate_text(lang_code, error)
        definitions[error] = [word_def, part_speech, new_translate]

    # score_def maps word:[common score, definition, part of speech, translated term]
    score_def = {}
    for word, common_score in frequency.items():
        for word1, definit in definitions.items():
            if word == word1:
                score_def[word] = [common_score, definit[0], definit[1], definit[2]["translatedText"]]


    context = {
        'score':score_user,
        'errors':errors,
        'user_input':mapping,
        'text':actual_sep,
        'card_dict': card_dict,
        'frequency': frequency,
        'mapping':mapping,
        'definitions': definitions, 
        'score_def': score_def
    }
        
    return render(request, "detect/recording_results.html", context)

# originalText = "Pretty soon I wanted to smoke, and asked the widow to let me.  But she wouldn't.  She said it was a mean practice and wasn't clean, and I must try to not do it any more.  That is just the way with some people.  They get down on a thing when they don't know nothing about it.  Here she was a-bothering about Moses, which was no kin to her, and no use to anybody, being gone, you see, yet finding a power of fault with me for doing a thing that had some good in it.  And she took snuff, too; of course that was all right, because she done it herself."
# transcriptText = "Pretty soon I wanted to smoke, and asked the window to let me.  But she wouldn't.  She said it was a mean practice and wasn't clean, and I must try to not do it any more.  That is just the way with some people.  They get down on a thing when they don't know anything about it.  Here she was a-bothering about Moses, which was no skin to her, and no use to nobody, being gone, you see, yet finding a power of fault with me for doing a thing that had some good in it.  And she took stuff, too; of course that was all right, because she done it herself."



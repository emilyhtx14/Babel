
var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition
var SpeechGrammarList = SpeechGrammarList || webkitSpeechGrammarList
var SpeechRecognitionEvent = SpeechRecognitionEvent || webkitSpeechRecognitionEvent

// var colors = [ 'aqua' , 'azure' , 'beige', 'bisque', 'black', 'blue', 'brown', 'chocolate', 'coral', 'crimson', 'cyan', 'fuchsia', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'indigo', 'ivory', 'khaki', 'lavender', 'lime', 'linen', 'magenta', 'maroon', 'moccasin', 'navy', 'olive', 'orange', 'orchid', 'peru', 'pink', 'plum', 'purple', 'red', 'salmon', 'sienna', 'silver', 'snow', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'white', 'yellow', 'open', 'stop'];
// var grammar = '#JSGF V1.0; grammar colors; public <color> = ' + colors.join(' | ') + ' ;'

var recognition = new SpeechRecognition();
var speechRecognitionList = new SpeechGrammarList();
// speechRecognitionList.addFromString(grammar, 1);
// recognition.grammars = speechRecognitionList;
recognition.continuous = false;
//list of languages: https://appmakers.dev/bcp-47-language-codes-list/
recognition.lang = 'en-US'; //'en-US' 'zh-CN'
recognition.interimResults = false;
recognition.maxAlternatives = 1;

var diagnostic = document.querySelector('.output');
var bg = document.querySelector('index1.html');
var hints = document.querySelector('.hints');

var colorHTML = '';
/*
colors.forEach(function(v, i, a){
  console.log(v, i);
  colorHTML += '<span style="background-color:' + v + ';"> ' + v + ' </span>';
});
*/

// hints.innerHTML = 'Tap/click then say a word to open two windows. Try ' + '\'Winter\' or \'Summer\'' + '.';
const voiceStart = document.getElementById('recordButton');
voiceStart.onclick = voice;

function voice() {
    recognition.start();
    console.log('Ready to receive a command.');
}

recognition.onresult = function (event) {
    // The SpeechRecognitionEvent results property returns a SpeechRecognitionResultList object
    // The SpeechRecognitionResultList object contains SpeechRecognitionResult objects.
    // It has a getter so it can be accessed like an array
    // The first [0] returns the SpeechRecognitionResult at the last position.
    // Each SpeechRecognitionResult object contains SpeechRecognitionAlternative objects that contain individual results.
    // These also have getters so they can be accessed like arrays.
    // The second [0] returns the SpeechRecognitionAlternative at position 0.
    // We then return the transcript property of the SpeechRecognitionAlternative object
    var color = event.results[0][0].transcript;
    // diagnostic.textContent = 'Result received: ' + color + '.';
    // bg.style.backgroundColor = color;


    console.log('The command:: -- ' + color);
    console.log('Confidence: ' + event.results[0][0].confidence);
    document.getElementById('recording-text-area').innerHTML = event.results[0][0].transcript;

    //document.getElementById("demo1").innerHTML = color;

}
/*
        recognition.onclick = function() {
          recognition.start();
          console.log('Ready to receive command ');
        }

 */
recognition.onspeechend = function () {
    console.log('onseechend: stop! ');
    recognition.stop();
}

recognition.onnomatch = function (event) {
    console.log('Invalid Color ');
    diagnostic.textContent = "I didn't recognise that color.";
}

recognition.onerror = function (event) {
    diagnostic.textContent = 'Error occurred in recognition: ' + event.error;
}
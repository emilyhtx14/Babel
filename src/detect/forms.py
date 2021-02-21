from django import forms
from .models import Languages, LANG_CHOICES, Record

class LanguageForm(forms.ModelForm):
    language = forms.ChoiceField(
        choices = LANG_CHOICES
    )
    class Meta:
        model = Languages
        fields = ['language']
        labels = {
                'language': 'lang',
            }


class RecordForm(forms.ModelForm):
    # true is actually default
    recording = forms.CharField(initial = "",
                               label='',
                            widget=forms.Textarea(
                                attrs={
                                    "id" : 'recording-text-area',
                                    "rows": 5,
                                    "cols": 70,
                                    "style": 'font-size: 20px',
                                }
                            ))
    class Meta:
        model = Record
        fields = ['recording']

       
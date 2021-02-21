from django.db import models

# Create your models here.

LANG_CHOICES = (
    ('af','Afrikaans'),
    ('sq', 'Albanian'),
    ('am','Amharic'),
    ('ar','Arabic'),
    ('hy','Armenian'),
    ('az','Azerbaijani'),
    ('eu','Basque'),
    ('be','Belarusian'),
    ('bn','Bengali'),
    ('bs','Bosnian'),
    ('bg','Bulgarian'),
    ('ca','Catalan'),
    ('ceb','Cebuano'),
    ('zh-TW','Chinese (Traditional)'),
    ('zh','Chinese (Simplified)'),
    ('co','Corsican'),
    ('hr','Croatian'),
    ('cs','Czech'),
    ('da','Danish'),
    ('nl','Dutch'),
    ('en','English'),
    ('eo','Esperanto'),
    ('et','Estonian'),
    ('fi','Finnish'),
    ('fr','French'),
    ('fy','Frisian'),
    ('de','German'),
    ('he','Hebrew'),
    ('it', 'Italian'),
    ('ja','Japanese'),
    ('ko', 'Korean'),
    ('la', 'Latin'),
    ('ms', 'Malay'),
    ('ny', 'Nynja'),
    ('fa','Persian'),
    ('ro', 'Romanian'),
    ('es', 'Spanish'),
    ('th','Thai'),
    ('uk', 'Ukrainian'),
    ('vi','Vietnamese'),
    ('cy','Welsh'),
    ('xh', 'Xhosa'),
    ('yi','Yiddish'),
    ('zu', 'Zulu')

)

class Languages(models.Model):
  language = models.CharField(max_length=20, choices=LANG_CHOICES)

class Record(models.Model):
  recording = models.CharField(max_length = 1000) 

class Texts(models.Model):
  passage = models.CharField(max_length = 1000)  
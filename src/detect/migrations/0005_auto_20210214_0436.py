# Generated by Django 3.1.6 on 2021-02-14 04:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detect', '0004_texts'),
    ]

    operations = [
        migrations.AlterField(
            model_name='languages',
            name='language',
            field=models.CharField(choices=[('af', 'Afrikaans'), ('sq', 'Albanian'), ('am', 'Amharic'), ('ar', 'Arabic'), ('hy', 'Armenian'), ('sv', 'Swedish'), ('zh', 'Chinese (Simplified)')], max_length=6),
        ),
        migrations.AlterField(
            model_name='record',
            name='recording',
            field=models.CharField(max_length=1000),
        ),
    ]
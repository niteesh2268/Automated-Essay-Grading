# Generated by Django 3.1.7 on 2021-04-22 21:37

from django.db import migrations
import django_base64field.fields


class Migration(migrations.Migration):

    dependencies = [
        ('grader', '0008_essay_min_score'),
    ]

    operations = [
        migrations.AddField(
            model_name='essay',
            name='image',
            field=django_base64field.fields.Base64Field(blank=True, default='', max_length=900000, null=True),
        ),
    ]
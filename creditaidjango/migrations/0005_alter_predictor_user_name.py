# Generated by Django 4.1.5 on 2023-01-26 09:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('creditaidjango', '0004_alter_predictor_ktppicture_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='predictor',
            name='user_name',
            field=models.CharField(default='no_name', max_length=255),
        ),
    ]

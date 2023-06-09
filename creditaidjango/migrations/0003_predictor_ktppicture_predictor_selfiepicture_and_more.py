# Generated by Django 4.1.5 on 2023-01-26 06:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('creditaidjango', '0002_predictor_created_at_predictor_updated_at'),
    ]

    operations = [
        migrations.AddField(
            model_name='predictor',
            name='ktppicture',
            field=models.ImageField(blank=True, null=True, upload_to='images/'),
        ),
        migrations.AddField(
            model_name='predictor',
            name='selfiepicture',
            field=models.ImageField(blank=True, null=True, upload_to='images/'),
        ),
        migrations.AlterField(
            model_name='predictor',
            name='user_name',
            field=models.CharField(default='no name', max_length=255),
        ),
    ]

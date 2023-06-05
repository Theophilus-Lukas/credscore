import pickle
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework import status
from .functions import general
from django.conf import settings
import numpy as np

from .models import Image
from .serializers import ImageSerializer

# import models
model_folder = str(settings.BASE_DIR) + "/creditaidjango/predictionmodels/"
result = {
    "status": status.HTTP_400_BAD_REQUEST,
    "data": "data"
}


@api_view(['GET'])
def ping(request):
    data_result = {'data': {'message': "Hello World"}}
    return JsonResponse(data_result, status=status.HTTP_200_OK)


@api_view(['GET'])
def predictHGBRPickle(request):
    data = request.data
    print(model_folder)
    return JsonResponse(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def predictHGBR(request):
    data = request.data
    return JsonResponse(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def predictHuber(request):
    data = request.data
    return JsonResponse(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def predictLGBM(request):
    data = request.data
    return JsonResponse(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def predictLassoReg(request):
    data = request.data

    Lasso_reg = pickle.load(open(model_folder+"lasso_reg_pickle.pkl", "rb"))
    input_data = [x for x in data.values()]
    final_features = [np.array(input_data)]

    result['data'] = Lasso_reg.predict(final_features)[0]
    result['status'] = status.HTTP_200_OK
    return JsonResponse(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def predictLasso(request):
    data = request.data
    return JsonResponse(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def predictPoisson(request):
    data = request.data
    return JsonResponse(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def get_prediction(request):
    predictor_id = request.query_params['id']
    try:
        predictor = Image.objects.get(id=predictor_id)
    except Image.DoesNotExist:
        return JsonResponse({'message': "PREDICTOR_NOT_FOUND"}, status=status.HTTP_404_NOT_FOUND)

    # ocr_result = oracle_v1.id_score(predictor_id)

    # prediction_result = {'data': {
    #     'result': ocr_result
    # }}
    return JsonResponse(result, status=status.HTTP_200_OK)


@api_view(['POST'])
def ocr_predictor(request):
    predictor_serializer = ImageSerializer(data=request.data)
    if (predictor_serializer.is_valid()):
        predictor_serializer.save()
        predictor_result = {'data': predictor_serializer.data}
    return JsonResponse(predictor_result, status=status.HTTP_201_CREATED)

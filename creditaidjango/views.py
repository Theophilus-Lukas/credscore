import pickle
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .functions import general
from django.conf import settings
import sklearn
import numpy as np

# import models
model_folder = str(settings.BASE_DIR) + "/creditaidjango/predictionmodels/"
result = {
    "status": status.HTTP_400_BAD_REQUEST,
    "data": "data"
}


@api_view(['GET'])
def ping(request):
    data_result = {'data': {'message': "Hello World"}}
    return Response(data_result, status=status.HTTP_200_OK)


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

    # Lasso = pickle.load(open(model_folder+"lasso.pkl", "rb"))
    # input_data = [x for x in data.values()]
    # final_features = [np.array(input_data)]

    # result['data'] = Lasso.predict(final_features)[0]
    # result['status'] = status.HTTP_200_OK
    return JsonResponse(result, status=status.HTTP_200_OK)


@api_view(['GET'])
def predictPoisson(request):
    data = request.data
    return JsonResponse(result, status=status.HTTP_200_OK)

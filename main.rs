
mod tensor;
// Указываем, что мы используем модуль `matrix`
//goto (исспользуй побольше
use ndarray::{arr2, Array, Array3, Axis, IxDyn};
use tensor::MyTensor;
/*
let dimensions = vec![2, 2]; // 2D тензор с размерностью 2x2
let mut tensor = MyTensor::new(dimensions.clone());//Создание пустого тензора
let tensor = MyTensor::from_data(result.into_dyn());//обычная матрица в tensor
tensor.fill(); // Заполняем тензор значениями
tensor.fill_random(); // Заполняем случайными значениями
let a = tensor.get_element(vec![1, 1]); //получаем элемент
tensor.set_element(vec![1, 1], 1.0); //Заменяем элемент
tensor.print_tensor(); //печать тензора
tensor.transpose(vec![1, 0]); //транспонирование
let result_tensor = tensor.multiply_scalar(10f64);//умножить на скаляр
let result_tensor = tensor.multiply_scalar(10f64);//прибавить скаляр
tensor.add_vector(&vector);//прибавить вектор
let result_tensor = tensor.add_tensor(&tensor);//Складывание тензоров
tensor.replace_layer(vec![ 1], new_layer.into_dyn()); //замена с настоящим тензором
tensor.add_layer(vec![ 1], new_layer.into_dyn()); //добавить чать тензора
tensor.multiply_adamar(vec![ 1], new_layer.into_dyn());//адамарово перемножение

*/
/*
{//Сложение тензора с самим собой
    let mut tensor = MyTensor::new(vec![3, 3]); // Создаем тензор 3x3
    tensor.fill(); // Заполняем его
    let tensor_clone = tensor.tensor.clone().into_dyn(); // Клонируем тензор и преобразуем в динамическую форму
    tensor.print_tensor();
    tensor.add_layer(vec![], tensor_clone); // Используем копию тензора

    // Печатаем тензор
    println!("Тензор после добавления слоя:");
    tensor.print_tensor();
}
{//сложение с обычным тензором
    let dimensions = vec![2,  3, 4, 4]; // 5D тензор
    let mut tensor = MyTensor::new(dimensions.clone());
    tensor.fill(); // Заполняем тензор значениями

    println!("Исходный 5D тензор:");
    tensor.print_tensor();

    // Создаем новый 3D слой для замены
    let new_layer = Array::from_shape_vec(
        (3, 4, 4), // 3D массив размерностью 3x4x4
        vec![
            // Первый слой
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            // Второй слой
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0,
            25.0, 26.0, 27.0, 28.0,
            29.0, 30.0, 31.0, 32.0,
            // Третий слой
            33.0, 34.0, 35.0, 36.0,
            37.0, 38.0, 39.0, 40.0,
            41.0, 42.0, 43.0, 44.0,
            45.0, 46.0, 47.0, 48.0
        ]
    ).unwrap();

    // Заменяем 2-й 3D слой в 5D тензоре новым 3D тензором
    tensor.replace_layer(vec![ 1], new_layer.into_dyn());

    println!("5D тензор после замены слоя:");
    tensor.print_tensor();

    let  mut a = ;
    a = ;
    println!("a: {}", a);
}
*/

fn main() {
    let matrix_4x4 = arr2(&[
        [2.0, 1.0, 4.0, 1.0],
        [3.0, 0.0, 1.0, 1.0],
        [-1.0, 2.0, 3.0, 4.0],
        [3.0, 1.0, 1.0, 1.0],
    ]);
    let tensor = MyTensor::from_data(matrix_4x4.into_dyn());
    println!("Детерминант 4x4: {}", tensor.determinant());
    let  a = 3;
}

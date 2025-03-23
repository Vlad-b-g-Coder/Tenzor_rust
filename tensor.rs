extern crate ndarray;

use ndarray::{Array, IxDyn, Dimension, ArrayBase, Data, OwnedRepr, Dim, Ix, ArrayD, s, Array1, Axis};
use rand::Rng;

pub struct MyTensor {
    pub tensor: Array<f64, IxDyn>,
}

impl MyTensor {


    pub fn new(dimensions: Vec<usize>) -> Self {
        let shape = IxDyn(&dimensions);
        MyTensor {
            tensor: Array::zeros(shape),
        }
    }
    pub fn from_data(data: ArrayD<f64>) -> Self {
        let shape = data.shape().to_vec();
        MyTensor {
            tensor: data.into_dyn(),
        }
    }
    pub fn to_ndarray(&self) -> ArrayD<f64> {
        self.tensor.clone()
    }
    pub fn fill(&mut self) {
        let shape = self.tensor.shape().to_owned();
        self.tensor = Array::from_shape_fn(shape, |indices| {
            // Convert `indices` to a vector manually
            let indices_vec: Vec<usize> = indices.slice().iter().map(|&i| i).collect();
            let value: usize = indices_vec.iter().sum();
            value as f64
        });
    }
    pub fn fill_random(&mut self) {
        let mut rng = rand::thread_rng(); // Создаем генератор случайных чисел
        let shape = self.tensor.shape().to_owned();
        self.tensor = Array::from_shape_fn(shape, |_| {
            let value: f64 = rng.gen_range(0.0..100.0); // Генерируем случайное число в диапазоне [0, 100)
            (value * 1.0).round() / 1.0 // Округляем до двух знаков после запятой
        });
    }
    pub fn get_element(&self, indices: Vec<usize>) -> f64 {
        // Convert Vec<usize> to &[usize]
        self.tensor[&indices[..]]
    }

    pub fn set_element(&mut self, indices: Vec<usize>, value: f64) {
        // Преобразование Vec<usize> в IxDyn для индексации
        let index = IxDyn(&indices);
        // Получение изменяемого элемента и установка нового значения
        if let Some(elem) = self.tensor.get_mut(index) {
            *elem = value;
        } else {
            panic!("Индекс выходит за пределы размерности тензора.");
        }
    }
    pub fn print_tensor(&self) {
        let dims = self.tensor.shape(); // Получаем размеры тензора
        let ndim = dims.len(); // Получаем количество осей (размерность)

        // Определяем функцию для печати тензора
        match ndim {
            1 => self.print_1d(),
            2 => self.print_2d(),
            3 => self.print_3d(),
            4 => self.print_4d(),
            5 => self.print_5d(),
            _ => println!("Тензор с размерностью {} не поддерживается.", ndim),
        }
    }
    fn print_1d(&self) {
        let data: Vec<f64> = self.tensor.iter().copied().collect();
        println!("{:?}", data);
    }
    fn print_2d(&self) {
        let (rows, cols) = (self.tensor.shape()[0], self.tensor.shape()[1]);
        for r in 0..rows {
            for c in 0..cols {
                print!("{:<4}", self.tensor[[r, c]]);
            }
            println!();
        }
    }
    fn print_3d(&self) {
        let (depth, rows, cols) = (self.tensor.shape()[0], self.tensor.shape()[1], self.tensor.shape()[2]);
        for d in 0..depth {
            println!("Слой {}:", d);
            for r in 0..rows {
                for c in 0..cols {
                    print!("{:<4}", self.tensor[[d, r, c]]);
                }
                println!();
            }
            println!(); // Пустая строка между слоями
        }
    }
    fn print_4d(&self) {
        let (batch, depth, rows, cols) = (
            self.tensor.shape()[0],
            self.tensor.shape()[1],
            self.tensor.shape()[2],
            self.tensor.shape()[3]
        );

        // Печать заголовков для пакетов
        for b in 0..batch {
            print!("Пакет {:<12}", b);
        }
        println!(); // Переход на следующую строку после заголовков

        // Печать данных
        for d in 0..depth {
            println!("Слой {}:", d);

            for r in 0..rows {
                for b in 0..batch {
                    // Печать строки для каждого пакета
                    for c in 0..cols {
                        print!("{:<4} ", self.tensor[[b, d, r, c]]);
                    }
                    print!("   "); // Пробелы между пакетами
                }
                println!(); // Переход на новую строку после всех пакетов
            }
            println!(); // Пустая строка между слоями
        }
    }
    fn print_5d(&self) {
    let (batch, depth, rows, cols, time) = (self.tensor.shape()[0], self.tensor.shape()[1], self.tensor.shape()[2], self.tensor.shape()[3], self.tensor.shape()[4]);
        for t in 0..time {
            println!("Временной шаг {}:", t);
            for b in 0..batch {
                println!("Пакет {}:", b);
                for d in 0..depth {
                    println!("Слой {}:", d);
                    for r in 0..rows {
                        for c in 0..cols {
                            print!("{:<4}", self.tensor[[b, d, r, c, t]]);
                        }
                        println!();
                    }
                    println!(); // Пустая строка между слоями
                }
                println!(); // Пустая строка между пакетами
            }
            println!(); // Пустая строка между временными шагами
        }
    }
    pub fn transpose(&mut self, axes: Vec<usize>) {
        let num_dims = self.tensor.ndim();

        if axes.len() != num_dims {
            panic!("Количество осей для транспонирования должно совпадать с количеством измерений тензора.");
        }

        let mut axis_set = std::collections::HashSet::new();
        for &axis in &axes {
            if axis >= num_dims {
                panic!("Ось {} выходит за пределы допустимого диапазона.", axis);
            }
            if !axis_set.insert(axis) {
                panic!("Каждая ось должна встречаться ровно один раз.");
            }
        }

        // Create a new tensor with permuted axes
        let permuted_tensor = self.tensor.clone().permuted_axes(axes);
        self.tensor = permuted_tensor;
    }
    pub fn multiply_scalar(&self, scalar: f64) -> MyTensor {
        let result_tensor = self.tensor.mapv(|x| x * scalar);
        MyTensor {
            tensor: result_tensor,
        }
    }
    pub fn add_vector(&self, vector: &Array1<f64>) -> MyTensor {
        // Получаем размерность тензора и вектора
        let tensor_shape = self.tensor.shape();
        let vector_len = vector.len();

        // Проверяем, что длина вектора соответствует размерности последнего измерения тензора
        if tensor_shape.last().unwrap() != &vector_len {
            panic!("Размер вектора не соответствует последнему измерению тензора.");
        }

        // Создаем новый тензор для хранения результата
        let mut result_tensor = self.tensor.clone();

        // Сложение вектора с тензором по последнему измерению
        for mut elem in result_tensor.axis_iter_mut(ndarray::Axis(tensor_shape.len() - 1)) {
            elem += vector;
        }

        MyTensor {
            tensor: result_tensor,
        }
    }
    pub fn add_tensor(&self, other: &MyTensor) -> MyTensor {
        if self.tensor.shape() != other.tensor.shape() {
            panic!("Формы тензоров не совпадают.");
        }
        let result_tensor = &self.tensor + &other.tensor;
        MyTensor {
            tensor: result_tensor.into_dyn(),
        }
    }
    pub fn replace_layer(&mut self, indices: Vec<usize>, new_layer: Array<f64, IxDyn>) {
        let tensor_shape = self.tensor.shape().to_vec();
        let new_layer_shape = new_layer.shape().to_vec();

        // Проверяем, что количество индексов плюс количество измерений нового слоя совпадает с количеством измерений тензора
        if indices.len() + new_layer_shape.len() != tensor_shape.len() {
            panic!("Количество индексов не соответствует размерности тензора.");
        }

        // Проверяем, что размеры нового слоя совпадают с соответствующими измерениями тензора
        for i in 0..new_layer_shape.len() {
            if tensor_shape[indices.len() + i] != new_layer_shape[i] {
                panic!("Размеры нового слоя не соответствуют текущему тензору.");
            }
        }

        // Рекурсивная функция для замены значений
        fn replace_recursively(
            tensor: &mut ArrayD<f64>,
            new_layer: &ArrayD<f64>,
            indices: &mut Vec<usize>,
            level: usize,
        ) {
            // Если достигнут последний уровень, то заменяем значение
            if level == new_layer.ndim() {
                tensor[indices.as_slice()] = new_layer[indices[(indices.len() - new_layer.ndim())..].as_ref()];
            } else {
                // Проходим по текущему измерению и углубляемся дальше
                for i in 0..new_layer.shape()[level] {
                    indices.push(i); // Добавляем индекс текущего измерения
                    replace_recursively(tensor, new_layer, indices, level + 1); // Рекурсивный вызов
                    indices.pop(); // Удаляем индекс после возврата
                }
            }
        }

        let mut full_indices = indices.clone();
        replace_recursively(&mut self.tensor, &new_layer, &mut full_indices, 0);
    }
    pub fn add_layer(&mut self, indices: Vec<usize>, new_layer: Array<f64, IxDyn>) {
        let tensor_shape = self.tensor.shape().to_vec();
        let new_layer_shape = new_layer.shape().to_vec();

        // Проверяем, что количество индексов плюс количество измерений нового слоя совпадает с количеством измерений тензора
        if indices.len() + new_layer_shape.len() != tensor_shape.len() {
            panic!("Количество индексов не соответствует размерности тензора.");
        }

        // Проверяем, что размеры нового слоя совпадают с соответствующими измерениями тензора
        for i in 0..new_layer_shape.len() {
            if tensor_shape[indices.len() + i] != new_layer_shape[i] {
                panic!("Размеры нового слоя не соответствуют текущему тензору.");
            }
        }

        // Рекурсивная функция для замены значений
        fn replace_recursively(
            tensor: &mut ArrayD<f64>,
            new_layer: &ArrayD<f64>,
            indices: &mut Vec<usize>,
            level: usize,
        ) {
            // Если достигнут последний уровень, то заменяем значение
            if level == new_layer.ndim() {
                tensor[indices.as_slice()] += new_layer[&indices[(indices.len() - new_layer.ndim())..]];
            } else {
                // Проходим по текущему измерению и углубляемся дальше
                for i in 0..new_layer.shape()[level] {
                    indices.push(i); // Добавляем индекс текущего измерения
                    replace_recursively(tensor, new_layer, indices, level + 1); // Рекурсивный вызов
                    indices.pop(); // Удаляем индекс после возврата
                }
            }
        }

        let mut full_indices = indices.clone();
        replace_recursively(&mut self.tensor, &new_layer, &mut full_indices, 0);
    }
    pub fn multiply_adamar(&mut self, indices: Vec<usize>, new_layer: Array<f64, IxDyn>) {
        let tensor_shape = self.tensor.shape().to_vec();
        let new_layer_shape = new_layer.shape().to_vec();

        // Проверяем, что количество индексов плюс количество измерений нового слоя совпадает с количеством измерений тензора
        if indices.len() + new_layer_shape.len() != tensor_shape.len() {
            panic!("Количество индексов не соответствует размерности тензора.");
        }

        // Проверяем, что размеры нового слоя совпадают с соответствующими измерениями тензора
        for i in 0..new_layer_shape.len() {
            if tensor_shape[indices.len() + i] != new_layer_shape[i] {
                panic!("Размеры нового слоя не соответствуют текущему тензору.");
            }
        }

        // Рекурсивная функция для замены значений
        fn replace_recursively(
            tensor: &mut ArrayD<f64>,
            new_layer: &ArrayD<f64>,
            indices: &mut Vec<usize>,
            level: usize,
        ) {
            // Если достигнут последний уровень, то заменяем значение
            if level == new_layer.ndim() {
                tensor[indices.as_slice()] *= new_layer[&indices[(indices.len() - new_layer.ndim())..]];
            } else {
                // Проходим по текущему измерению и углубляемся дальше
                for i in 0..new_layer.shape()[level] {
                    indices.push(i); // Добавляем индекс текущего измерения
                    replace_recursively(tensor, new_layer, indices, level + 1); // Рекурсивный вызов
                    indices.pop(); // Удаляем индекс после возврата
                }
            }
        }

        let mut full_indices = indices.clone();
        replace_recursively(&mut self.tensor, &new_layer, &mut full_indices, 0);
    }




    // Метод для вычисления детерминанта 3x3 матрицы
    fn det_3x3(matrix: &Array<f64, IxDyn>) -> f64 {
        let a = matrix[[0, 0]];
        let b = matrix[[0, 1]];
        let c = matrix[[0, 2]];

        let det_minor_1 = matrix[[1, 1]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 1]];
        let det_minor_2 = matrix[[1, 0]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 0]];
        let det_minor_3 = matrix[[1, 0]] * matrix[[2, 1]] - matrix[[1, 1]] * matrix[[2, 0]];

        a * det_minor_1 - b * det_minor_2 + c * det_minor_3
    }

    // Метод для вычисления детерминанта 4x4 матрицы с пошаговым выводом
    fn det_4x4(matrix: &Array<f64, IxDyn>) -> f64 {
        let mut expression = String::new();
        let mut determinant = 0.0;
        let mut minor_strs = vec![];

        for col in 0..4 {
            let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
            let minor = MyTensor::minor(matrix, 0, col);
            let det_minor = MyTensor::det_3x3(&minor);
            let term = sign * matrix[[0, col]] * det_minor;

            // Форматирование минора в строки
            let minor_str = format!(
                "| {} {} {} |\n\
                 | {} {} {} |\n\
                 | {} {} {} |",
                minor[[0, 0]], minor[[0, 1]], minor[[0, 2]],
                minor[[1, 0]], minor[[1, 1]], minor[[1, 2]],
                minor[[2, 0]], minor[[2, 1]], minor[[2, 2]]
            );

            let minor_lines: Vec<String> = minor_str.lines().map(|s| s.to_string()).collect();
            minor_strs.push(minor_lines);

            if col > 0 {
                expression.push_str(" + ");
            }

            expression.push_str(&format!(
                "{} * (-1)^({}+{}) * ",
                matrix[[0, col]], 1, col + 1
            ));
        }

        // Создаем строки для вывода
        let mut formatted_expression = String::new();
        let num_lines = minor_strs[0].len();

        for i in 0..num_lines {
            let mut line_parts = vec![];
            for (col, minor_lines) in minor_strs.iter().enumerate() {
                line_parts.push(minor_lines[i].clone());  // Преобразуем в String
                if col < minor_strs.len() - 1 {
                    line_parts.push(" ".repeat(10));  // Используем String для пробелов
                }
            }
            formatted_expression.push_str(&line_parts.concat());
            formatted_expression.push_str("\n");
        }

        formatted_expression.push_str(&expression);

        println!("{}\n= {}", formatted_expression, determinant);

        determinant
    }

    // Метод для получения минора матрицы
    fn minor(matrix: &Array<f64, IxDyn>, row: usize, col: usize) -> Array<f64, IxDyn> {
        let shape = matrix.shape();
        let mut minor_shape = shape.to_vec();
        minor_shape[0] -= 1;
        minor_shape[1] -= 1;

        let mut minor = Array::zeros(IxDyn(&minor_shape));
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                if i != row && j != col {
                    let minor_i = if i > row { i - 1 } else { i };
                    let minor_j = if j > col { j - 1 } else { j };
                    minor[[minor_i, minor_j]] = matrix[[i, j]];
                }
            }
        }
        minor
    }

    // Основной метод для вычисления детерминанта
    pub fn determinant(&self) -> f64 {
        let shape = self.tensor.shape();
        match shape {
            &[4, 4] => MyTensor::det_4x4(&self.tensor),
            _ => panic!("Детерминант вычисляется только для матриц размером 4x4"),
        }
    }




}

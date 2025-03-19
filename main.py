import numpy as np
import pandas as pd


def get_restrictions_matr(columns_names, data_base):
    """ Функция создаёт матрицу ограничений по указанным столбцам базы данных """
    restrictions = np.array([])
    for col in columns_names:
        restrictions = np.append(restrictions, data_base[col].values)

    return restrictions.reshape(len(columns_names), -1)


def get_new_restrictions_matr(restrictions):
    """ Функция изменяет матрицу коэффициентов, учитывая диапазон значений """
    # Расширяем
    n_rows = len(restrictions)
    restrictions = np.pad(restrictions, ((0, n_rows), (0, 0)), mode='constant', constant_values=0)
    # Дублируем начальные коэффициенты
    n_rows = len(restrictions)
    for row in range(n_rows - 1, -1, -2):
        restrictions[row] = restrictions[row // 2]
        restrictions[row - 1] = restrictions[row // 2]

    # Создаем матрицу с диагональю с чередующимися +1 и -1
    diagonal = np.array([1 if i % 2 == 0 else -1 for i in range(n_rows)])
    matrix = np.diag(diagonal)

    # Объединяем нашу матрицу с диагональной
    return np.hstack((restrictions, matrix))


def check_zero_delta(restrictions_matr, vector_b):
    """ Функция изменяет матрицу ограничений для строк, где дельта была 0 """
    len_b = len(vector_b)
    i = 0
    while i < len_b:
        if vector_b[i] == vector_b[i + 1]:
            restrictions_matr = np.delete(restrictions_matr, i + 1, axis=0)

            restrictions_matr = np.delete(restrictions_matr, -(len_b - i), axis=1)
            restrictions_matr = np.delete(restrictions_matr, -(len_b - i - 1), axis=1)

            vector_b = np.delete(vector_b, i + 1)

            i -= 1
            len_b -= 1
        i += 2
    return restrictions_matr, list(vector_b)


def add_count_restrictions(z_str, b_vect, rest_matr, count_x, max_count):
    # Меняем z
    z_str = np.append(z_str, [0 for i in range(count_x)])
    # Меняем b
    b_vect.extend([max_count] * count_x)
    # Меняем r
    zeros_columns = np.zeros((rest_matr.shape[0], count_x))
    rest_matr = np.hstack((rest_matr, zeros_columns))
    new_rows = np.zeros((count_x, rest_matr.shape[1]))
    for i in range(new_rows.shape[0]):
        new_rows[i, i] = 1
        new_rows[i, new_rows.shape[1] - count_x + i] = 1
    rest_matr = np.vstack((rest_matr, new_rows))

    return z_str, b_vect, rest_matr


def find_m(restrictions):
    """ Функция подсчитывает параметр m-метода, в зависимости от значений в симплекс таблице """
    eps = 3
    return restrictions.max() ** eps


def find_position_of_unit(simplex_table, column):
    """ Функция ищет позицию единицы в столбце. Используется для базисных столбцов. """
    return np.where(simplex_table[:, column] == 1)[0][0]


def create_simplex_table(restrictions, solution_vector_b, z_coefficients):
    """ Функция создаёт симплекс таблицу, по заданным параметрам """
    n_rows, n_columns = restrictions.shape

    simplex_table = np.zeros((n_rows + 1, n_columns + 1))
    for row in range(n_rows):
        simplex_table[row] = np.append(restrictions[row], solution_vector_b[row])

    simplex_table[n_rows] = z_coefficients

    return simplex_table


def delete_basis_var_from_z_string(simplex_table, basis_place_mass):
    """ Функция изменяет z-строку так, чтобы в ней не было базисных переменных """
    n_rows = len(simplex_table[:, 0]) - 1

    for basis_place in basis_place_mass:
        if simplex_table[n_rows][basis_place] != 0:
            pos_one = find_position_of_unit(simplex_table, basis_place)
            simplex_table[n_rows] = simplex_table[n_rows] - simplex_table[pos_one] * simplex_table[n_rows][basis_place]

    return simplex_table


def get_corner_point(simplex_table, basis_place_mass):
    """ Функция находит угловую точку """
    n_columns = len(simplex_table[0]) - 1

    corner_x = np.zeros(n_columns)
    for basis_place in basis_place_mass:
        pos_one = find_position_of_unit(simplex_table, basis_place)
        corner_x[basis_place] = simplex_table[pos_one, n_columns]

    return corner_x


def change_basis_set(simplex_table, basis_place_mass):
    """ Функция убирает одну переменную из базиса и заменяет её на одну из свободных, конкретным способом """
    n_columns = len(simplex_table[0]) - 1
    n_rows = len(simplex_table[:, 0]) - 1

    # Находим разрешающий столбец (столбец с максимальным значением в z-строке)
    solution_vector_pos = np.argmax(simplex_table[n_rows][:n_columns])

    # Находим разрешающую строку (строка с минимальным положительным отношением),
    solution_vector = simplex_table[:n_rows, solution_vector_pos]
    b_vector = simplex_table[:n_rows, n_columns]
    mask = solution_vector > 0
    # Проверка на наличие решения
    if len(mask) == 0:
        print("Решений нет!")
        exit(1)

    positive_ratios = np.full_like(b_vector, np.inf, dtype=np.float64)  # Предварительно заполняем inf
    # Выполняем деление только для положительных элементов
    positive_ratios[mask] = b_vector[mask] / solution_vector[mask]
    solution_string_pos = np.argmin(positive_ratios)

    # Найденный элемент является разрешающим
    solution_elem = simplex_table[solution_string_pos, solution_vector_pos]

    # Обновляем список базисных переменных
    # Поиск в решающей строке базисный вектор, у которого значение 1
    for i in range(len(basis_place_mass)):
        if simplex_table[solution_string_pos][basis_place_mass[i]] == 1:
            basis_place_mass[i] = solution_vector_pos
            break

    # Обновляем симплекс-таблицу
    new_simplex_table = np.zeros_like(simplex_table)
    # Делим разрешающую строку на разрешающий элемент
    new_simplex_table[solution_string_pos] = simplex_table[solution_string_pos] / solution_elem
    for row in range(n_rows + 1):
        if row != solution_string_pos:
            # Обновляем остальные строки
            factor = simplex_table[row, solution_vector_pos]
            new_simplex_table[row] = simplex_table[row] - factor * new_simplex_table[solution_string_pos]

    simplex_table = new_simplex_table

    return simplex_table


def find_basis(restrictions, solution_vector_b, z_coefficients,
               count_basis_var, basis_place_mass, simplex_table):
    """ Функция выполняет работу M-метода, ищет базис в начальной задаче """
    m = find_m(restrictions)

    n_rows, n_columns = restrictions.shape
    # Меняем ограничения
    count_new_col = n_rows - count_basis_var  # кол-во столбцов, которые надо добавить
    # Получаем массив готовых строк
    pos_one_mass = []
    for basis_place in basis_place_mass:
        pos_one_mass.append(int(find_position_of_unit(simplex_table, basis_place)))
    new_restrictions = np.pad(restrictions, ((0, 0), (0, count_new_col)), mode='constant', constant_values=0)
    # Добавляем базисные столбцы, для строк не содержащихся в pos_one_mass
    pos_one = 0
    for new_col in range(n_columns, n_columns + count_new_col):
        while pos_one in pos_one_mass:
            pos_one += 1
        new_restrictions[pos_one, new_col] = 1
        basis_place_mass.append(new_col)
        pos_one += 1
    restrictions = new_restrictions

    # Меняем z-строку (Штрафуем функцию)
    fine_mass = [-m] * count_new_col
    fine_mass.append(0)
    z_coefficients = np.append(z_coefficients[:n_columns], fine_mass)

    # Создаём новую симплекс таблицу
    n_rows, n_columns = restrictions.shape
    simplex_table = create_simplex_table(restrictions, solution_vector_b, z_coefficients)

    # Убираем базисные переменные из z-строки
    simplex_table = delete_basis_var_from_z_string(simplex_table, basis_place_mass)

    iteration = 2
    while iteration - 2 < count_new_col * 3:
        # Проверяем, остались ли штрафные переменные в угловой точке
        corner_x = get_corner_point(simplex_table, basis_place_mass)
        if np.all(corner_x[n_columns - count_new_col:] == 0) and all(num not in basis_place_mass for num in range(n_columns - count_new_col, n_columns)):
            return np.delete(simplex_table, range(n_columns - count_new_col, n_columns, 1), axis=1)

        simplex_table = change_basis_set(simplex_table, basis_place_mass)
        iteration += 1
    return 1


def m_method(restrictions, solution_vector_b, z_coefficients):
    """
    M-метод минимизирует функцию с коэффициентами z_coefficients с ограничениями restrictions = solution_vector_b
    Параметры:
    - restrictions: Матрица ограничений (каждая строка - ограничение).
    - solution_vector_b: Вектор значений правой части ограничений.
    - z_coefficients: Коэффициенты целевой функции и начальное значение (z-строка в симплекс таблице).
    """
    n_rows, n_columns = restrictions.shape

    simplex_table = create_simplex_table(restrictions, solution_vector_b, z_coefficients)

    # Инициализируем список для позиций базисных переменных
    basis_place_mass = []
    for col in range(n_columns):
        # Проверяем, является ли столбец базисным
        column = simplex_table[:n_rows, col]  # Проверяем только строки ограничений
        if np.count_nonzero(column) == 1 and np.sum(column) == 1:
            basis_place_mass.append(col)

    # Ищем из претендентов на базис, реальные базисные переменные
    count_basis_var = len(basis_place_mass)
    if count_basis_var >= n_rows:
        new_basis_place_mass = []
        for i in range(count_basis_var - 1):
            is_basis_col = True
            for j in range(i + 1, count_basis_var):
                if np.array_equal(simplex_table[:n_rows, i], simplex_table[:n_rows, j]):
                    is_basis_col = False
                    break
            if is_basis_col:
                new_basis_place_mass.append(basis_place_mass[i])
        new_basis_place_mass.append(basis_place_mass[count_basis_var - 1])
        basis_place_mass = new_basis_place_mass

    # Проверяем есть ли явный базис
    is_explicit_basis = True
    count_basis_var = len(basis_place_mass)
    if count_basis_var < n_rows:
        is_explicit_basis = False

    # Если явного базиса нет, то симплекс метод не доступен
    if not is_explicit_basis:
        simplex_table = find_basis(restrictions, solution_vector_b, z_coefficients, count_basis_var,
                                   basis_place_mass, simplex_table)
        if isinstance(simplex_table, int) and simplex_table == 1:
            return 1, 1
    simplex_table = delete_basis_var_from_z_string(simplex_table, basis_place_mass)

    while True:
        # Проверяем, является ли текущая z-строка оптимальной (для задачи минимизации)
        is_optimal = np.all(simplex_table[n_rows][:n_columns] <= 0)

        corner_x = get_corner_point(simplex_table, basis_place_mass)

        if is_optimal:
            z_min = simplex_table[n_rows, n_columns]
            return np.round(corner_x, 2), z_min

        simplex_table = change_basis_set(simplex_table, basis_place_mass)


def find_non_integer_solution(sol_x):
    """ Функция находит нецелый x из решения и его позицию"""
    for i, x in enumerate(sol_x):
        if not isinstance(x, int) and not (isinstance(x, float) and x.is_integer()):
            return x, i
    return None, -1


def get_new_tasks(curr_x, pos_x, restrictions, solution_vector_b, z_coefficients):
    """ Функция создаёт 2 новых ветви для curr_x """
    # Новый вектор b
    left_solution_vector_b = np.append(solution_vector_b, int(curr_x))
    right_solution_vector_b = np.append(solution_vector_b, int(curr_x) + 1)

    # Новая z строка
    new_z_coefficients = np.append(z_coefficients, 0)

    # Новая матрица
    zeros_column = np.zeros((restrictions.shape[0], 1))
    new_restrictions = np.hstack((restrictions, zeros_column))
    new_left_str = np.zeros(len(new_restrictions[0]))
    new_left_str[pos_x] = 1
    new_left_str[-1] = 1
    new_right_str = np.copy(new_left_str)
    new_right_str[-1] = -1
    left_restrictions = np.copy(new_restrictions)
    left_restrictions = np.vstack((left_restrictions, new_left_str))
    right_restrictions = np.copy(new_restrictions)
    right_restrictions = np.vstack((right_restrictions, new_right_str))

    return left_solution_vector_b, right_solution_vector_b, new_z_coefficients, left_restrictions, right_restrictions


def process_single_task(task, tasks_mass, opposite_tasks_mass, task_number, solution_mass, current_best_res):
    """ Функция решает ветвь, добавляет новые ветви или добавляет границу """
    sol_x, sol_z = m_method(task[0], task[1], task[2])
    # Проверка на отсутствие решения
    if isinstance(sol_x, int) and sol_x == 1 and sol_z == 1:
        tasks_mass[task_number] = None
        return current_best_res
    # Поиск нецелого решения
    curr_x, pos_x = find_non_integer_solution(sol_x)
    if not curr_x and pos_x == -1:
        # Обновляем лучшего решения, если оно лучше текущего
        if current_best_res is None or sol_z < current_best_res[1]:
            current_best_res = (sol_x, sol_z)
        tasks_mass[task_number] = None
        return current_best_res

    # Генерация новых задач
    left_solution_vector_b, right_solution_vector_b, new_z_coefficients, left_restrictions, right_restrictions = (
        get_new_tasks(curr_x, pos_x, task[0], task[1], task[2]))

    # Добавление решений и новых задач
    solution_mass.append((sol_x, sol_z))
    opposite_tasks_mass.append((right_restrictions, right_solution_vector_b, new_z_coefficients))
    tasks_mass.append((left_restrictions, left_solution_vector_b, new_z_coefficients))

    tasks_mass[task_number] = None
    return current_best_res


def modern_m_method(restrictions, solution_vector_b, z_coefficients):
    """ Функция ищет целочисленное решение поставленной задачи """
    right_tasks_mass, left_tasks_mass, solution_mass = [], [], []

    # Решаем задачу M-методом
    sol_x, sol_z = m_method(restrictions, solution_vector_b, z_coefficients)
    if isinstance(sol_x, int) and sol_x == 1 and sol_z == 1:
        raise ValueError("При данных ограничениях решения нет")

    # Добавляем решение в массив решений
    solution_mass.append((sol_x, sol_z))

    # Находим НЕ целый x
    curr_x, pos_x = find_non_integer_solution(sol_x)
    if not curr_x and pos_x == -1:
        print("Все x целые")
        return sol_x, sol_z

    # Составляем новые задачи и добавляем их в массив задач
    left_solution_vector_b, right_solution_vector_b, new_z_coefficients, left_restrictions, right_restrictions = (
        get_new_tasks(curr_x, pos_x, restrictions, solution_vector_b, z_coefficients))

    # Добавляем новые задачи в массивы задач
    right_tasks_mass.append((right_restrictions, right_solution_vector_b, new_z_coefficients))
    left_tasks_mass.append((left_restrictions, left_solution_vector_b, new_z_coefficients))

    iteration = 0
    res = None
    while True:
        # 1. Выбор задачи
        min_z = float('inf')
        task_number = -1
        for i, solution in enumerate(solution_mass):
            curr_z = solution[1]
            if curr_z < min_z and (left_tasks_mass[i] or right_tasks_mass[i]) and (not res or curr_z < (res[1] - 1)):
                task_number = i
                min_z = curr_z
        if task_number == -1:
            print("Все задачи решены, количество итераций -", iteration)
            return res

        # Решаем левую задачу, если она доступна
        left_task = left_tasks_mass[task_number]
        if left_task:
            res = process_single_task(left_task, left_tasks_mass, right_tasks_mass, task_number, solution_mass, res)
        else:
            # Решаем правую
            right_task = right_tasks_mass[task_number]
            res = process_single_task(right_task, right_tasks_mass, left_tasks_mass, task_number, solution_mass, res)

        iteration += 1
        if iteration > 2000:
            if res:
                print("Слишком много итераций!")
                return res
            else:
                print("Слишком много итераций!")
                exit(1)


def restaurant_optimization(*columns_names, data_name, optimization_column):
    """
    Функция устанавливает параметры из базы данных и запускает м-метод
    Структура базы данных должна быть такой, что первый столбец - название, второй - оптимизируемый столбец
    """
    # Устанавливаем опции печати
    np.set_printoptions(suppress=True)
    # Устанавливаем опции для отображения всех строк и столбцов
    pd.set_option('display.max_columns', None)  # Показывать все столбцы

    # Получаем базу данных
    data = pd.read_csv(data_name)

    # Получаем данные из БД
    restrictions_matr = get_restrictions_matr(columns_names, data_base=data)
    len_database = len(restrictions_matr[0])
    # Получаем вектор b
    solution_vector_b_start = []
    print(f"Введите максимальное отклонение ограничений в процентах %:", end=" ")
    delta_percent = float(input())
    print("Введите ограничения:")
    for condition in range(len(restrictions_matr[:, 0])):
        print(f"{data.columns[condition + 2]}:", end=" ")
        b_vector_value = float(input())
        delta = (b_vector_value * delta_percent) / 100
        solution_vector_b_start.append(b_vector_value + delta)
        solution_vector_b_start.append(b_vector_value - delta)

    restrictions_matr = get_new_restrictions_matr(restrictions_matr)
    restrictions_matr, solution_vector_b_start = check_zero_delta(restrictions_matr, solution_vector_b_start)

    print("Введите максимальное количество 1ого блюда: ", end="")
    max_count = int(input())

    print()
    # Получаем тип оптимизации
    print("Какой тип оптимизации использовать?")
    print("1 - min\n2 - max")
    while (optimization_type := input()) not in ["1", "2"]:
        print("Введите цифру:\n1 - min\n2 - max")

    # Получаем z-строку и запускаем м-метод
    if optimization_type == "1":
        z_str = np.append(data[optimization_column].values * (-1), [0 for row in range(len(restrictions_matr[0]) - len_database + 1)])
        z_str, solution_vector_b_start, restrictions_matr = add_count_restrictions(z_str, solution_vector_b_start,
                                                                                   restrictions_matr, len_database, max_count)
        x_sol, z_sol = modern_m_method(restrictions_matr, solution_vector_b_start, z_str)
    else:
        z_str = np.append(data[optimization_column].values, [0 for row in range(len(restrictions_matr[0]) - len_database + 1)])
        z_str, solution_vector_b_start, restrictions_matr = add_count_restrictions(z_str, solution_vector_b_start,
                                                                                   restrictions_matr, len_database, max_count)
        x_sol, z_sol = modern_m_method(restrictions_matr, solution_vector_b_start, z_str)
        z_sol *= (-1)

    x_sol = x_sol[:len_database]

    # Выводим решение
    non_zero_indices = np.nonzero(x_sol)
    sol_data = pd.DataFrame(index=[i + 1 for i in non_zero_indices])
    sol_data[data.columns[0]] = data[data.columns[0]].values[non_zero_indices]
    sol_data["Количество"] = x_sol[non_zero_indices]
    for col_name in data.columns[1:]:
        sol_data[col_name] = data[col_name].values[non_zero_indices] * x_sol[non_zero_indices]
    print("\nОптимальное решение в задаче:")
    print(sol_data)
    print(f"Общая цена = {z_sol}")


restaurant_optimization(
    "Белки (г)", "Жиры (г)", "Углеводы (г)",
    data_name="food_data.csv",
    optimization_column="Цена (руб)"
)

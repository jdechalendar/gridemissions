# import pytest

from gridemissions.eia_api import column_name_to_ba

key = "EBA.%s-ALL.D.H"
ba = "MISO"
column_name = key % ba
test_data = [(column_name, key, ba)]

key = "EBA.%s-%s.ID.H"
ba1 = "MISO"
ba2 = "PJM"
ba = f"{ba1}-{ba2}"
column_name = key % (ba1, ba2)
test_data.append((column_name, key, ba))

# @pytest.mark.parametrize("column_name,key,ba", test_data, ids=range(len(test_data)))
def test_column_name_to_ba(column_name, key, ba):
    assert column_name_to_ba(column_name, key) == ba


if __name__ == "__main__":
    for column_name, key, ba in test_data:
        print(column_name)
        test_column_name_to_ba(column_name, key, ba)

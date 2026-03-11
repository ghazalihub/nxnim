import nim_nx
let a = tensor([7i32], s32)
echo "popcount: ", a.population_count()
echo "clz: ", tensor([8i32], s32).count_leading_zeros()
